// Blocked LU with partial pivoting, real and complex (Linalg::factor_lu / factor_lu_complex, LUFactor::solve, determinant,
// leading_determinants, schur) against an independent MPFR replay of the documented rounding sequence (docs/numerics.md, "LU
// factorization", "Complex LU and Pfaffians", "Pivoting segments and partial elimination"), bit for bit. The reference is
// left-looking, column by column: column j receives the U rows and trailing update of each earlier block (reference::dot_sub on
// interleaved terms), then its in-block dots, the pivot search within its segment and the divisions (MPFR); whole rows are swapped
// as pivots are chosen. Also: GPU and host updates agree; leading determinants equal separate calls on A[0:b, 0:b]; det(A) =
// det(leading) det(Schur); complex |det|^2 equals the determinant of the real embedding and solves agree with complex QR.
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include "element_reference.hpp"
#include <iostream>
#include <numeric>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<class R> struct RefLU {std::vector<typename R::T> W,lead;typename R::T det;std::vector<std::size_t> perm;int sign=1;std::size_t zero_pivot,status_column;};
template<class R> RefLU<R> reference_lu(std::vector<typename R::T> W,std::size_t n,std::size_t nb,const std::vector<std::size_t>& bounds,std::size_t ke){
    using T=typename R::T;RefLU<R> r;r.perm.resize(n);std::iota(r.perm.begin(),r.perm.end(),0);r.zero_pivot=r.status_column=n;
    std::vector<std::size_t> limits=bounds;if(ke<n)limits.push_back(ke);
    auto seg=[&](std::size_t j){for(auto b:limits)if(b>j)return b;return n;};
    // Block starts: multiples of nb, plus ke.
    std::vector<std::size_t> starts;for(std::size_t k=0;k<ke;k=std::min({n,k+nb,ke}))starts.push_back(k);starts.push_back(ke);
    auto at=[&](std::size_t i,std::size_t j)->T&{return W[i*n+j];};
    auto D=[&](std::size_t i,std::size_t j,std::size_t q0,std::size_t q1){std::vector<T> x,y;for(std::size_t q=q0;q<q1;++q){x.push_back(at(i,q));y.push_back(at(q,j));}return R::D(at(i,j),x,y);};
    // Column j (j < ke) or a trailing column (j >= ke): the U rows and trailing updates of the blocks before min(j, ke)'s block.
    auto block_start=[&](std::size_t j){std::size_t k0=0;for(auto s:starts)if(s<=j)k0=s;return k0;};
    auto earlier=[&](std::size_t j){const std::size_t limit=j<ke?block_start(j):ke;
        for(std::size_t b=0;b+1<starts.size()&&starts[b+1]<=limit;++b){const std::size_t b0=starts[b],b1=starts[b+1];
            for(std::size_t i=b0;i<b1;++i)at(i,j)=D(i,j,b0,i);for(std::size_t i=b1;i<n;++i)at(i,j)=D(i,j,b0,b1);}};
    T prod=R::one();bool first=true;std::size_t poison=n,bi=0;r.lead.assign(bounds.size(),R::zeros(invalid));std::size_t stop=n;
    for(std::size_t j=0;j<ke;++j){const std::size_t k0=block_start(j);
        earlier(j);for(std::size_t i=k0;i<j;++i)at(i,j)=D(i,j,k0,i);for(std::size_t i=j;i<n;++i)at(i,j)=D(i,j,k0,j);
        const std::size_t e=seg(j);word st=0;std::size_t p=j;for(std::size_t i=j;i<e;++i){st|=R::status(at(i,j));if(R::compare(at(i,j),at(p,j))>0)p=i;}
        if(st){stop=r.status_column=j;break;}
        if(p!=j){for(std::size_t c=0;c<n;++c)std::swap(at(p,c),at(j,c));std::swap(r.perm[p],r.perm[j]);r.sign=-r.sign;}
        prod=first?at(j,j):R::product(prod,at(j,j));first=false;
        for(;bi<bounds.size()&&bounds[bi]==j+1;++bi)if(j+1<=poison)r.lead[bi]=r.sign<0?R::neg(prod):prod;
        if(R::is_zero(at(j,j))){if(r.zero_pivot==n)r.zero_pivot=j;poison=std::min(poison,e);continue;}
        for(std::size_t i=j+1;i<n;++i)at(i,j)=R::quotient(at(i,j),at(j,j));}
    if(stop<n){for(std::size_t i=0;i<n;++i)for(std::size_t c=stop;c<n;++c)at(i,c)=R::zeros(invalid);}
    else for(std::size_t j=ke;j<n;++j)earlier(j);   // trailing columns: the U rows and the Schur complement
    r.det=stop<n||ke>poison?R::zeros(invalid):r.sign<0?R::neg(prod):prod;if(!ke&&stop==n)r.det=R::one();
    r.W=std::move(W);return r;
}
template<class R> std::vector<typename R::T> reference_solve(const RefLU<R>& f,const std::vector<typename R::T>& Bm,std::size_t n,std::size_t nrhs,std::size_t nb){
    using T=typename R::T;std::vector<T> X(n*nrhs);const auto& W=f.W;
    for(std::size_t i=0;i<n;++i)for(std::size_t c=0;c<nrhs;++c)X[i*nrhs+c]=Bm[f.perm[i]*nrhs+c];
    auto D=[&](T v,std::size_t i,std::size_t c,std::size_t q0,std::size_t q1){std::vector<T> x,y;for(std::size_t q=q0;q<q1;++q){x.push_back(W[i*n+q]);y.push_back(X[q*nrhs+c]);}return R::D(v,x,y);};
    for(std::size_t c=0;c<nrhs;++c){
        for(std::size_t i=0;i<n;++i){const std::size_t r0=i/nb*nb;T v=X[i*nrhs+c];for(std::size_t b0=0;b0<r0;b0+=nb)v=D(v,i,c,b0,b0+nb);X[i*nrhs+c]=D(v,i,c,r0,i);}
        for(std::size_t i=n;i-->0;){const std::size_t r0=i/nb*nb,r1=std::min(n,r0+nb);T v=X[i*nrhs+c];
            for(std::size_t b0=(n-1)/nb*nb;b0>r0;b0-=nb)v=D(v,i,c,b0,std::min(n,b0+nb));
            if(i+1<r1)v=D(v,i,c,i+1,r1);X[i*nrhs+c]=R::quotient(v,W[i*n+i]);}}
    return X;
}
template<class R> std::vector<typename R::T> random_matrix(std::size_t rows,std::size_t cols,std::mt19937_64& rng,int span){
    std::vector<typename R::T> A(rows*cols);for(auto& x:A){x=R::rnd(rng,span);if(rng()%13==0)x=R::zeros();}return A;}
template<int B,class R> LUFactor factor(Linalg& la,const std::vector<typename R::T>& A,std::size_t n,const EliminationOptions& o){
    return R::complex?la.factor_lu_complex(B,A.data(),n,o):la.factor_lu(B,A.data(),n,o);}
template<int B,class R> void check(Linalg& la,const std::vector<typename R::T>& A,std::size_t n,std::size_t nb,const std::vector<std::size_t>& bounds,std::size_t stop,
                                   const std::string& what,std::mt19937_64& rng){
    using T=typename R::T;EliminationOptions o;o.block=nb;o.host_macs=0;o.solve_host_macs=0;o.bounds=bounds;o.stop_at=stop;EliminationOptions h=o;h.gpu=false;
    const std::size_t blk=nb?std::min(nb,std::max<std::size_t>(n,1)):std::max<std::size_t>(n,1),ke=stop?stop:n;
    auto ref=reference_lu<R>(A,n,blk,bounds,ke);auto f=factor<B,R>(la,A,n,o),g=factor<B,R>(la,A,n,h);
    const T* W=static_cast<const T*>(f.lu());const T* V=static_cast<const T*>(g.lu());
    for(std::size_t p=0;p<n*n;++p){require(R::equal(W[p],ref.W[p]),what+": factor entry "+std::to_string(p)+" != MPFR replay");require(same(W[p],V[p]),what+": GPU and host factors differ");}
    for(std::size_t i=0;i<n;++i)require(f.permutation()[i]==ref.perm[i],what+": permutation");
    require(f.permutation_sign()==ref.sign&&f.info().zero_pivot==ref.zero_pivot&&f.info().status_column==ref.status_column&&f.eliminated()==ke,what+": info");
    if(ke>nb&&nb&&ref.status_column==n)require(f.info().gpu_updates>0,what+": no GPU update");
    T det;f.determinant(&det);require(R::equal(det,ref.det),what+": determinant");
    std::vector<T> lead(bounds.size());f.leading_determinants(lead.data());for(std::size_t b=0;b<bounds.size();++b)require(R::equal(lead[b],ref.lead[b]),what+": leading determinant "+std::to_string(b));
    if(ke<n){std::vector<T> S((n-ke)*(n-ke));f.schur(S.data());for(std::size_t i=ke;i<n;++i)for(std::size_t j=ke;j<n;++j)require(same(S[(i-ke)*(n-ke)+j-ke],W[i*n+j]),what+": schur copy");}
    else{const std::size_t nrhs=3;auto Bm=random_matrix<R>(n,nrhs,rng,4);std::vector<T> X(n*nrhs),Y(n*nrhs);f.solve(Bm.data(),nrhs,X.data());g.solve(Bm.data(),nrhs,Y.data());
        if(ref.status_column==n){auto want=reference_solve<R>(ref,Bm,n,nrhs,blk);
            for(std::size_t p=0;p<X.size();++p){require(R::equal(X[p],want[p]),what+": solve entry "+std::to_string(p)+" != MPFR replay");require(same(X[p],Y[p]),what+": GPU and host solves differ");}}
        else for(auto& x:X)require(R::status(x)==invalid&&R::is_zero(x),what+": solve of a status factor");}
}
// Leading determinants equal separate calls on A[0:b, 0:b] (bounds truncated to b), bit for bit.
template<int B,class R> void separate(Linalg& la,const std::vector<typename R::T>& A,std::size_t n,std::size_t nb,const std::vector<std::size_t>& bounds,const std::string& what){
    using T=typename R::T;EliminationOptions o;o.block=nb;o.host_macs=0;o.bounds=bounds;auto f=factor<B,R>(la,A,n,o);std::vector<T> lead(bounds.size());f.leading_determinants(lead.data());
    for(std::size_t k=0;k<bounds.size();++k){const std::size_t b=bounds[k];std::vector<T> sub(b*b);for(std::size_t i=0;i<b;++i)for(std::size_t j=0;j<b;++j)sub[i*b+j]=A[i*n+j];
        EliminationOptions s=o;s.bounds.assign(bounds.begin(),bounds.begin()+long(k)+1);T d;factor<B,R>(la,sub,b,s).determinant(&d);
        require(same(d,lead[k]),what+": leading determinant "+std::to_string(k)+" differs from a separate call");}
}
template<int B> bool close(const Float<B>& a,const Float<B>& b){auto d=reference::real<B>(Operation::sub,a,b);return !d.sign||d.exponent<std::max(a.exponent,b.exponent)-(B-40);}
template<int B,class R> void accuracy(Linalg& la,std::mt19937_64& rng,const std::string& what){
    using T=typename R::T;const std::size_t n=48,k=20;auto A=random_matrix<R>(n,n,rng,0);   // near the identity: I + entries of about 2^-6
    auto small=[](Float<B>& x){if(x.sign)x.exponent-=6;};
    for(auto& x:A){if constexpr(R::complex){small(x.re);small(x.im);}else small(x);}
    for(std::size_t i=0;i<n;++i){Float<B>& d=[&]()->Float<B>&{if constexpr(R::complex)return A[i*n+i].re;else return A[i*n+i];}();d=reference::real<B>(Operation::add,d,RR<B>::one());}
    EliminationOptions o;o.block=8;T det,lead,sdet;factor<B,R>(la,A,n,o).determinant(&det);
    EliminationOptions s=o;s.stop_at=k;auto f=factor<B,R>(la,A,n,s);f.determinant(&lead);std::vector<T> S((n-k)*(n-k));f.schur(S.data());factor<B,R>(la,S,n-k,o).determinant(&sdet);
    const T prod=R::product(lead,sdet);
    if constexpr(R::complex)require(close<B>(prod.re,det.re)&&close<B>(prod.im,det.im),what+": det(A) != det(leading) det(Schur)");else require(close<B>(prod,det),what+": det(A) != det(leading) det(Schur)");
    if constexpr(R::complex){
        // |det A|^2 = det [[Re, -Im], [Im, Re]]; solves agree with complex QR.
        std::vector<Float<B>> E(4*n*n);for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){const T& a=A[i*n+j];
            E[i*2*n+j]=a.re;E[i*2*n+n+j]=negate(a.im);E[(n+i)*2*n+j]=a.im;E[(n+i)*2*n+n+j]=a.re;}
        Float<B> ed;la.factor_lu(B,E.data(),2*n,o).determinant(&ed);const auto m2=reference::real<B>(Operation::add,reference::real<B>(Operation::mul,det.re,det.re),reference::real<B>(Operation::mul,det.im,det.im));
        require(close<B>(m2,ed),what+": |det|^2 != det of the real embedding");
        const std::size_t nrhs=4;auto Bm=random_matrix<R>(n,nrhs,rng,0);std::vector<T> X(n*nrhs),Q(n*nrhs);la.factor_lu_complex(B,A.data(),n,o).solve(Bm.data(),nrhs,X.data());
        la.factor_qr_complex(B,A.data(),n,n).solve(Bm.data(),nrhs,Q.data());
        for(std::size_t p=0;p<X.size();++p)require(close<B>(X[p].re,Q[p].re)&&close<B>(X[p].im,Q[p].im),what+": LU and QR solutions differ");}
}
template<int B,class R> void suite(const char* label){
    std::mt19937_64 rng(11+B+R::complex);Linalg la;const std::string w=std::string(label)+" bits="+std::to_string(B);
    for(std::size_t n:{0,1,2,5,37,70})for(std::size_t nb:{0,1,8,16,32}){
        auto A=random_matrix<R>(n,n,rng,n%2?6:300);check<B,R>(la,A,n,nb,{},0,w+" n="+std::to_string(n)+" nb="+std::to_string(nb),rng);}
    // Segments: leading determinants, bounds 1 apart, b = n; partial elimination at a block end and inside a block.
    {const std::size_t n=60;auto A=random_matrix<R>(n,n,rng,6);
        for(std::size_t nb:{0,8,16}){check<B,R>(la,A,n,nb,{5,6,20,33,60},0,w+" bounds nb="+std::to_string(nb),rng);separate<B,R>(la,A,n,nb,{5,6,20,33,60},w+" bounds");
            check<B,R>(la,A,n,nb,{7,24},24,w+" stop_at 24 nb="+std::to_string(nb),rng);check<B,R>(la,A,n,nb,{},37,w+" stop_at 37 nb="+std::to_string(nb),rng);}}
    {const std::size_t n=40;auto S=random_matrix<R>(n,n,rng,6);[&]()->word&{if constexpr(R::complex)return S[25*n+30].im.status;else return S[25*n+30].status;}()=exponent_overflow;check<B,R>(la,S,n,16,{10,30},0,w+" status entry",rng);
        auto Z=random_matrix<R>(n,n,rng,6);for(std::size_t r=0;r<n;++r)Z[r*n+9]=R::zeros();check<B,R>(la,Z,n,8,{},0,w+" zero column",rng);
        auto Y=random_matrix<R>(n,n,rng,6);for(std::size_t r=0;r<12;++r)Y[r*n+9]=R::zeros();check<B,R>(la,Y,n,8,{12,20,40},0,w+" zero pivot in a segment",rng);
        auto f=factor<B,R>(la,Z,n,EliminationOptions{});require(!f.nonsingular()&&f.info().zero_pivot==9,w+": zero column not reported as zero pivot 9");}
    accuracy<B,R>(la,rng,w);
    std::cout<<w<<": LU factors, permutations, solves, determinants, leading determinants and Schur complements match the MPFR replay\n";
}
int main(){try{suite<256,RR<256>>("real");suite<1152,RR<1152>>("real");suite<256,RC<256>>("complex");suite<1152,RC<1152>>("complex");return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
