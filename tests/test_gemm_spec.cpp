// Strided, signed and triangular products (Linalg::gemm with a GemmSpec, syr2k, block utilities; docs/numerics.md, "Strided,
// signed and triangular products"): submatrix views with offsets and leading dimensions, both transposes, overwrite/add/subtract,
// lower/upper output with the rest of C untouched, on the GPU path, the forced host path and resident Buffer operands, against
// MPFR dots (reference::dot) with one rounding per entry. Usage: test_wolfnum_gemm_spec
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include <cstring>
#include <iostream>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<class E,class Fn> void rejects(Fn f,const std::string& what){bool ok=false;try{f();}catch(const E&){ok=true;}require(ok,"not rejected: "+what);}
template<class T> bool same(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<int B> using F=Float<B>;
template<int B> F<B> with_exponent(F<B> x,int e){x.exponent=e;return x;}
// Random entries; some lines get a second band, a spread fallback, zeros or a status entry.
template<int B> std::vector<F<B>> matrix(std::size_t rows,std::size_t cols,std::mt19937_64& rng){
    std::vector<F<B>> A(rows*cols);
    for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){F<B> x=reference::random_number<B>(rng,6);
        if(rng()%11==0)x=zero<B/32>();if(r%5==1&&c%3==0)x.exponent-=150;
        if(r%7==3&&c<3)x=with_exponent<B>(reference::random_number<B>(rng,6),-4000);A[r*cols+c]=x;}
    A[(rows/2)*cols+cols/2].status=invalid;return A;
}
// RN(c + s * dot) with one rounding: overwrite (c unused), add (s = +1) and subtract (s = -1).
template<int B> F<B> expected(GemmUpdate u,F<B> c,const std::vector<F<B>>& x,const std::vector<F<B>>& y){
    if(u==GemmUpdate::overwrite)return reference::dot<B>(x.data(),1,y.data(),1,x.size());
    if(u==GemmUpdate::subtract)return reference::dot_sub<B>(c,x.data(),1,y.data(),1,x.size());
    return negate(reference::dot_sub<B>(negate(c),x.data(),1,y.data(),1,x.size()));
}
bool written(Triangle t,std::size_t i,std::size_t j){return t==Triangle::full||(t==Triangle::lower?i>=j:i<=j);}
const char* name(GemmUpdate u){return u==GemmUpdate::overwrite?"overwrite":u==GemmUpdate::add?"add":"subtract";}
// Stored operands are views (offset, leading dimension) into larger arrays; C's untouched entries keep their sentinel bits.
template<int B> struct Case {
    std::size_t m,n,k,lda,ldb,ldc;GemmSpec spec;std::vector<F<B>> A,Bm,C0;
    Case(std::size_t m_,std::size_t n_,std::size_t k_,bool ta,bool tb,GemmUpdate u,Triangle t,std::mt19937_64& rng):m(m_),n(n_),k(k_){
        spec.transpose_a=ta;spec.transpose_b=tb;spec.update=u;spec.triangle=t;
        const std::size_t ar=ta?k:m,ac=ta?m:k,br=tb?n:k,bc=tb?k:n;lda=ac+3;ldb=bc+2;ldc=n+5;
        spec.lda=lda;spec.ldb=ldb;spec.ldc=ldc;spec.a_offset=2*lda+1;spec.b_offset=ldb+3;spec.c_offset=3*ldc+2;
        A=matrix<B>(ar+3,lda,rng);Bm=matrix<B>(br+2,ldb,rng);C0=matrix<B>(m+4,ldc,rng);C0[spec.c_offset].status=0;}
    F<B> a(std::size_t i,std::size_t q)const{return A[spec.a_offset+(spec.transpose_a?q*lda+i:i*lda+q)];}
    F<B> b(std::size_t q,std::size_t j)const{return Bm[spec.b_offset+(spec.transpose_b?j*ldb+q:q*ldb+j)];}
    void check(const std::vector<F<B>>& C,const std::string& what)const{
        std::size_t bad=0;std::string first;
        for(std::size_t p=0;p<C.size();++p){const std::size_t rel=p-spec.c_offset,i=rel/ldc,j=rel%ldc;
            const bool in=p>=spec.c_offset&&i<m&&j<n&&written(spec.triangle,i,j);F<B> want=C0[p];
            if(in){std::vector<F<B>> x(k),y(k);for(std::size_t q=0;q<k;++q){x[q]=a(i,q);y[q]=b(q,j);}want=expected<B>(spec.update,C0[p],x,y);}
            if(!(in?reference::equal<B>(C[p],want):same(C[p],want))&&!bad++)first=(in?"entry ":"untouched entry ")+std::to_string(p);}
        require(!bad,what+": "+std::to_string(bad)+" mismatches, first "+first);}
};
template<int B> void host_products(Linalg& la,const char* label,std::mt19937_64& rng){
    for(bool ta:{false,true})for(bool tb:{false,true})for(GemmUpdate u:{GemmUpdate::overwrite,GemmUpdate::add,GemmUpdate::subtract})
        for(Triangle t:{Triangle::full,Triangle::lower,Triangle::upper}){
            const std::size_t m=t==Triangle::full?7:9;Case<B> c(m,9,11,ta,tb,u,t,rng);auto C=c.C0;
            la.gemm(B,c.A.data(),c.Bm.data(),c.m,c.n,c.k,C.data(),c.spec);
            const auto& r=la.report();require(la.options().force_fallback?!r.gpu_outputs:r.gpu_outputs&&r.multi_band_outputs&&r.fallback_outputs,"placement mix not exercised: gpu="+std::to_string(r.gpu_outputs)+" multi="+std::to_string(r.multi_band_outputs)+" fallback="+std::to_string(r.fallback_outputs));
            c.check(C,std::string(label)+" bits="+std::to_string(B)+" ta="+std::to_string(ta)+" tb="+std::to_string(tb)+" "+name(u)+" tri="+std::to_string(int(t)));}
    // The default spec is the plain gemm, bit for bit; K = 0 leaves an update unchanged and overwrites with zero.
    {auto A=matrix<B>(6,5,rng),Bm=matrix<B>(5,4,rng);std::vector<F<B>> C1(24),C2(24);
        la.gemm(B,false,A.data(),Bm.data(),6,4,5,C1.data());la.gemm(B,A.data(),Bm.data(),6,4,5,C2.data(),GemmSpec{});
        for(std::size_t p=0;p<24;++p)require(same(C1[p],C2[p]),"default spec differs from gemm");
        GemmSpec s;s.update=GemmUpdate::add;auto C3=C1;la.gemm(B,A.data(),Bm.data(),6,4,0,C3.data(),s);for(std::size_t p=0;p<24;++p)require(same(C1[p],C3[p]),"empty-K add changed C");}
    rejects<std::invalid_argument>([&]{GemmSpec s;s.triangle=Triangle::lower;std::vector<F<B>> x(12);la.gemm(B,x.data(),x.data(),3,4,1,x.data(),s);},"triangle with m != n");
    rejects<std::invalid_argument>([&]{GemmSpec s;s.lda=2;std::vector<F<B>> x(12);la.gemm(B,x.data(),x.data(),3,4,3,x.data(),s);},"lda below k");
}
template<int B> void syr2k_cases(Linalg& la,std::mt19937_64& rng){
    const std::size_t n=10,k=6;
    for(bool rows:{false,true})for(bool skew:{false,true})for(Triangle t:{Triangle::full,Triangle::lower})for(GemmUpdate u:{GemmUpdate::overwrite,GemmUpdate::subtract}){
        const std::size_t cols=rows?k:n,lines=rows?n:k,ld=cols+2;auto A=matrix<B>(lines+1,ld,rng),Bm=matrix<B>(lines+1,ld,rng),C0=matrix<B>(n,n+1,rng);
        GemmSpec s;s.transpose_a=rows;s.lda=s.ldb=ld;s.ldc=n+1;s.a_offset=ld;s.b_offset=1;s.update=u;s.triangle=t;
        auto el=[&](const std::vector<F<B>>& M,std::size_t off,std::size_t i,std::size_t q){return M[off+(rows?i*ld+q:q*ld+i)];};
        auto C=C0;la.syr2k(B,A.data(),Bm.data(),n,k,C.data(),skew,s);
        for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n+1;++j){const std::size_t p=i*(n+1)+j;
            if(j==n||!written(t,i,j)){require(same(C[p],C0[p]),"syr2k touched an entry outside the output");continue;}
            std::vector<F<B>> x(2*k),y(2*k);
            for(std::size_t q=0;q<k;++q){x[q]=el(A,s.a_offset,i,q);x[k+q]=el(Bm,s.b_offset,i,q);y[q]=el(Bm,s.b_offset,j,q);y[k+q]=el(A,s.a_offset,j,q);if(skew)y[k+q]=negate(y[k+q]);}
            require(reference::equal<B>(C[p],expected<B>(u,C0[p],x,y)),"syr2k bits="+std::to_string(B)+" rows="+std::to_string(rows)+" skew="+std::to_string(skew)+" entry "+std::to_string(p));}}
}
template<int B> void utilities(Linalg& la,std::mt19937_64& rng){
    constexpr int N=B/32;const std::size_t r=7,c=5,ld=8;auto X=matrix<B>(c+1,ld,rng),Y=matrix<B>(r,ld,rng);
    auto Z=Y;la.copy_block(B,X.data(),ld,Z.data(),ld,5,c,false,true);
    for(std::size_t i=0;i<5;++i)for(std::size_t j=0;j<ld;++j)require(same(Z[i*ld+j],j<c?negate(X[i*ld+j]):Y[i*ld+j]),"copy_block negate");
    Z=Y;la.copy_block(B,X.data(),ld,Z.data(),ld,r,c-1,true);  // Y (7 x 4) = X^T for X (4 x 7)
    for(std::size_t i=0;i<r;++i)for(std::size_t j=0;j<ld;++j)require(same(Z[i*ld+j],j<c-1?X[j*ld+i]:Y[i*ld+j]),"copy_block transpose");
    Z=Y;la.negate_block(B,Z.data(),r,c,ld);for(std::size_t i=0;i<r;++i)for(std::size_t j=0;j<ld;++j)require(same(Z[i*ld+j],j<c?negate(Y[i*ld+j]):Y[i*ld+j]),"negate_block");
    F<B> alpha=reference::random_number<B>(rng,4);Z=Y;la.axpy_block(B,&alpha,X.data(),ld,Z.data(),ld,c,c);
    for(std::size_t i=0;i<c;++i)for(std::size_t j=0;j<c;++j){F<B> w=fma<N>(alpha,X[i*ld+j],Y[i*ld+j]);require(reference::equal<B>(Z[i*ld+j],w),"axpy_block");}
}
template<int B> void resident(Engine& e,std::mt19937_64& rng){
    Linalg la(e);
    for(GemmUpdate u:{GemmUpdate::overwrite,GemmUpdate::add,GemmUpdate::subtract})for(Triangle t:{Triangle::full,Triangle::upper})for(bool tb:{false,true}){
        Case<B> c(9,9,11,u==GemmUpdate::add,tb,u,t,rng);
        auto Ab=e.make_buffer<F<B>>(c.A.size()),Bb=e.make_buffer<F<B>>(c.Bm.size()),Cb=e.make_buffer<F<B>>(c.C0.size());
        Ab.upload(c.A.data(),c.A.size());Bb.upload(c.Bm.data(),c.Bm.size());Cb.upload(c.C0.data(),c.C0.size());
        auto batch=e.batch();auto ticket=la.gemm(batch,Ab,Bb,c.m,c.n,c.k,Cb,c.spec);batch.submit().wait();require(ticket.resolved(),"resident ticket");
        std::vector<F<B>> C(c.C0.size());Cb.download(C.data(),C.size());
        c.check(C,"resident bits="+std::to_string(B)+" "+name(u)+" tri="+std::to_string(int(t))+" tb="+std::to_string(tb));}
    rejects<std::invalid_argument>([&]{auto x=e.make_buffer<F<B>>(16);auto y=e.make_buffer<F<B>>(16);auto z=e.make_buffer<F<B>>(16);GemmSpec s;s.require_gpu=true;
        auto batch=e.batch();la.gemm(batch,x,y,2,2,2,z,s);},"resident require_gpu");
}
template<int B> void width(Engine& e){
    std::mt19937_64 rng(77+B);Linalg gpu;LinalgOptions forced;forced.force_fallback=true;Linalg host(forced);
    host_products<B>(gpu,"gpu",rng);host_products<B>(host,"host",rng);syr2k_cases<B>(gpu,rng);syr2k_cases<B>(host,rng);utilities<B>(gpu,rng);resident<B>(e,rng);
    std::cout<<B<<" bits: strided/transposed/signed/triangular gemm, syr2k and block utilities match MPFR\n";
}
int main(){try{Engine e;width<256>(e);width<1152>(e);return 0;}catch(const std::exception& x){std::cerr<<x.what()<<'\n';return 1;}}
