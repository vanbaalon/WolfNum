// Pfaffians of real and complex skew-symmetric matrices (Linalg::pfaffian / pfaffian_complex; docs/numerics.md, "Pfaffian",
// "Pivoting segments and partial elimination") against an independent MPFR replay of the documented blocked Parlett-Reid
// sequence, bit for bit. The reference keeps the full skew matrix with its original entries and applies every block's terms
// lazily when an entry is read (one dot_sub per earlier block, then the in-block dot), exchanging whole rows and columns; the
// library applies earlier blocks eagerly on lower storage. Also: GPU and host agree; leading Pfaffians equal separate calls on
// A[0:b, 0:b]; Pf(A) = Pf(leading) Pf(Schur); Pf(A)^2 = det(A) (factor_lu / factor_lu_complex) to working accuracy.
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include "element_reference.hpp"
#include <iostream>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<class R> struct RefPf {typename R::T value;std::vector<typename R::T> lead,schur;std::size_t zero_pivot,status_column;};
template<class R> RefPf<R> reference_pfaffian(const std::vector<typename R::T>& A,std::size_t n,std::size_t block,const std::vector<std::size_t>& bounds,std::size_t stop){
    using T=typename R::T;RefPf<R> r;r.zero_pivot=r.status_column=n;r.lead.assign(bounds.size(),R::zeros(invalid));
    const bool odd=!stop&&n%2;const std::size_t ke=stop?stop:n-odd;
    auto oddvalue=[&]{word st=0;for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j)st|=R::status(A[i*n+j]);return R::zeros(st);};
    r.value=R::one();if(odd&&bounds.empty()){r.value=oddvalue();return r;}
    const std::size_t nb=block?std::min(std::max<std::size_t>(ke,2),(block+1)/2*2):std::max<std::size_t>(ke+ke%2,2);
    std::vector<std::size_t> limits=bounds;if(ke<n)limits.push_back(ke);auto seg=[&](std::size_t k){for(auto b:limits)if(b>k)return b;return n;};
    std::vector<T> F(n*n,R::zeros());for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j){F[i*n+j]=A[i*n+j];F[j*n+i]=R::neg(A[i*n+j]);}
    std::vector<std::vector<T>> P(n),Q(n);std::vector<std::size_t> sizes;   // per index: the terms of every finished block, then this block's
    auto value=[&](std::size_t i,std::size_t c,std::size_t inblock){T v=F[i*n+c];std::size_t off=0;
        for(auto K:sizes){v=R::D(v,std::vector<T>(P[i].begin()+long(off),P[i].begin()+long(off+K)),std::vector<T>(Q[c].begin()+long(off),Q[c].begin()+long(off+K)));off+=K;}
        return R::D(v,std::vector<T>(P[i].begin()+long(off),P[i].begin()+long(off+inblock)),std::vector<T>(Q[c].begin()+long(off),Q[c].begin()+long(off+inblock)));};
    for(auto& v:P)v.assign(n+2*nb,R::zeros());for(auto& v:Q)v.assign(n+2*nb,R::zeros());
    bool first=true;int sign=1;std::size_t bi=0,off=0;
    auto leads=[&](std::size_t upto,const T& v){for(;bi<bounds.size()&&bounds[bi]<=upto;++bi)r.lead[bi]=v;};
    auto done=[&]{if(odd)r.value=oddvalue();return r;};
    for(std::size_t k0=0;k0<ke;){const std::size_t k1=std::min({n,k0+nb,ke});std::size_t s=0;
        for(std::size_t k=k0;k<k1;k+=2,++s){
            std::vector<T> x(n);word st=0;std::size_t p=k+1;const std::size_t e=seg(k);
            for(std::size_t i=k+1;i<n;++i)x[i]=value(i,k,2*s);
            for(std::size_t i=k+1;i<e;++i){st|=R::status(x[i]);if(R::compare(x[i],x[p])>0)p=i;}
            if(st){r.status_column=k;r.value=R::zeros(st);return done();}
            if(R::is_zero(x[p])){r.zero_pivot=k;leads(e,R::zeros());r.value=e>=ke?R::zeros():R::zeros(invalid);return done();}
            if(p!=k+1){const std::size_t a=k+1;for(std::size_t c=0;c<n;++c)std::swap(F[a*n+c],F[p*n+c]);for(std::size_t i=0;i<n;++i)std::swap(F[i*n+a],F[i*n+p]);
                std::swap(P[a],P[p]);std::swap(Q[a],Q[p]);std::swap(x[a],x[p]);sign=-sign;}
            const T a=R::neg(x[k+1]);r.value=first?a:R::product(r.value,a);first=false;leads(k+2,sign<0?R::neg(r.value):r.value);
            for(std::size_t i=k+2;i<n;++i){const T y=value(i,k+1,2*s),tau=R::quotient(x[i],a);
                P[i][off+2*s]=tau;P[i][off+2*s+1]=y;Q[i][off+2*s]=y;Q[i][off+2*s+1]=R::neg(tau);}
        }
        sizes.push_back(2*s);off+=2*s;k0=k1;}
    if(sign<0)r.value=R::neg(r.value);
    if(stop&&ke<n){const std::size_t m=n-ke;r.schur.assign(m*m,R::zeros());
        for(std::size_t i=0;i<m;++i)for(std::size_t j=0;j<i;++j){const T v=value(ke+i,ke+j,0);r.schur[i*m+j]=v;r.schur[j*m+i]=R::neg(v);}}
    return done();
}
template<class R> std::vector<typename R::T> skew(std::size_t n,std::mt19937_64& rng,int span){
    std::vector<typename R::T> A(n*n);for(auto& x:A)x=R::rnd(rng,span); // upper part is arbitrary: never read
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j)if(rng()%11==0)A[i*n+j]=R::zeros();return A;}
template<int B,class R> PfaffianInfo pf(Linalg& la,const std::vector<typename R::T>& A,std::size_t n,typename R::T* out,const EliminationOptions& o,
                                        typename R::T* lead=nullptr,typename R::T* schur=nullptr){
    return R::complex?la.pfaffian_complex(B,A.data(),n,out,o,lead,schur):la.pfaffian(B,A.data(),n,out,o,lead,schur);}
template<int B,class R> void check(Linalg& la,const std::vector<typename R::T>& A,std::size_t n,std::size_t nb,const std::vector<std::size_t>& bounds,std::size_t stop,const std::string& what){
    using T=typename R::T;EliminationOptions o;o.block=nb;o.host_macs=0;o.bounds=bounds;o.stop_at=stop;EliminationOptions h=o;h.gpu=false;
    T g,c;const std::size_t m=stop?n-stop:0;std::vector<T> lg(bounds.size()),lh(bounds.size()),sg(m*m),sh(m*m);
    auto gi=pf<B,R>(la,A,n,&g,o,lg.data(),sg.data()),hi=pf<B,R>(la,A,n,&c,h,lh.data(),sh.data());auto ref=reference_pfaffian<R>(A,n,nb,bounds,stop);
    require(R::equal(g,ref.value),what+": Pfaffian != MPFR replay");require(same(g,c),what+": GPU and host differ");
    require(gi.zero_pivot==ref.zero_pivot&&gi.status_column==ref.status_column&&gi.sign==hi.sign,what+": info");
    for(std::size_t b=0;b<bounds.size();++b){require(R::equal(lg[b],ref.lead[b]),what+": leading Pfaffian "+std::to_string(b)+" != MPFR replay");require(same(lg[b],lh[b]),what+": GPU and host leading values differ");}
    for(std::size_t p=0;p<m*m;++p){require(R::equal(sg[p],ref.schur[p]),what+": Schur entry "+std::to_string(p)+" != MPFR replay");require(same(sg[p],sh[p]),what+": GPU and host Schur complements differ");}
    const std::size_t ke=stop?stop:n,blk=nb?(nb+1)/2*2:n;if(ke%2==0&&ke>blk&&ref.status_column==n&&ref.zero_pivot==n)require(gi.gpu_updates>0,what+": no GPU update");
}
template<int B,class R> void separate(Linalg& la,const std::vector<typename R::T>& A,std::size_t n,std::size_t nb,const std::vector<std::size_t>& bounds,const std::string& what){
    using T=typename R::T;EliminationOptions o;o.block=nb;o.host_macs=0;o.bounds=bounds;T v;std::vector<T> lead(bounds.size());pf<B,R>(la,A,n,&v,o,lead.data());
    for(std::size_t k=0;k<bounds.size();++k){const std::size_t b=bounds[k];std::vector<T> sub(b*b);for(std::size_t i=0;i<b;++i)for(std::size_t j=0;j<b;++j)sub[i*b+j]=A[i*n+j];
        EliminationOptions s=o;s.bounds.assign(bounds.begin(),bounds.begin()+long(k)+1);T d;pf<B,R>(la,sub,b,&d,s);
        require(same(d,lead[k]),what+": leading Pfaffian "+std::to_string(k)+" differs from a separate call");}
}
template<int B> bool close(const Float<B>& a,const Float<B>& b){auto d=reference::real<B>(Operation::sub,a,b);return !d.sign||d.exponent<std::max(a.exponent,b.exponent)-(B-40);}
template<int B,class R> bool close_t(const typename R::T& a,const typename R::T& b){if constexpr(R::complex)return close<B>(a.re,b.re)&&close<B>(a.im,b.im);else return close<B>(a,b);}
template<int B,class R> void accuracy(Linalg& la,std::mt19937_64& rng,const std::string& what){
    using T=typename R::T;const std::size_t n=36,k=14;auto A=skew<R>(n,rng,2);EliminationOptions o;o.block=8;T v,lead,sv;
    pf<B,R>(la,A,n,&v,o);std::vector<T> full(n*n,R::zeros());for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j){full[i*n+j]=A[i*n+j];full[j*n+i]=R::neg(A[i*n+j]);}
    T det;(R::complex?la.factor_lu_complex(B,full.data(),n,o):la.factor_lu(B,full.data(),n,o)).determinant(&det);
    require(close_t<B,R>(R::product(v,v),det),what+": Pf^2 != det");
    EliminationOptions s=o;s.stop_at=k;std::vector<T> S((n-k)*(n-k));pf<B,R>(la,A,n,&lead,s,nullptr,S.data());pf<B,R>(la,S,n-k,&sv,o);
    require(close_t<B,R>(R::product(lead,sv),v),what+": Pf(A) != Pf(leading) Pf(Schur)");
}
template<int B,class R> void suite(const char* label){
    using T=typename R::T;std::mt19937_64 rng(23+B+R::complex);Linalg la;const std::string w=std::string(label)+" bits="+std::to_string(B);
    for(std::size_t n:{0,1,2,3,4,7,10,36,64,71})for(std::size_t nb:{0,2,3,8,16,32}){
        auto A=skew<R>(n,rng,n%3?5:200);check<B,R>(la,A,n,nb,{},0,w+" n="+std::to_string(n)+" nb="+std::to_string(nb));}
    {const std::size_t n=60;auto A=skew<R>(n,rng,5);
        for(std::size_t nb:{0,8,16}){check<B,R>(la,A,n,nb,{2,4,20,34,60},0,w+" bounds nb="+std::to_string(nb));separate<B,R>(la,A,n,nb,{2,4,20,34,60},w+" bounds");
            check<B,R>(la,A,n,nb,{6,24},24,w+" stop_at 24 nb="+std::to_string(nb));check<B,R>(la,A,n,nb,{},38,w+" stop_at 38 nb="+std::to_string(nb));}
        auto O=skew<R>(61,rng,5);check<B,R>(la,O,61,8,{10,30,60},0,w+" odd n with bounds");}
    {const std::size_t n=40;auto S=skew<R>(n,rng,5);[&]()->word&{if constexpr(R::complex)return S[33*n+20].re.status;else return S[33*n+20].status;}()=exponent_overflow;
        check<B,R>(la,S,n,8,{10,30},0,w+" status entry");
        auto Z=skew<R>(n,rng,5);for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j)if(i==12||j==12)Z[i*n+j]=R::zeros();check<B,R>(la,Z,n,8,{},0,w+" zero row and column");
        check<B,R>(la,Z,n,8,{6,12,14,20,40},0,w+" zero pivot in a segment");
        T v;auto info=pf<B,R>(la,Z,n,&v,EliminationOptions{});require(R::is_zero(v)&&!R::status(v)&&info.zero_pivot<n,w+": zero column: Pf != 0 or no zero pivot");}
    accuracy<B,R>(la,rng,w);
    std::cout<<w<<": Pfaffians, leading Pfaffians and Schur complements match the MPFR replay; GPU and host agree; Pf^2 = det\n";
}
int main(){try{suite<256,RR<256>>("real");suite<1152,RR<1152>>("real");suite<256,RC<256>>("complex");suite<1152,RC<1152>>("complex");return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
