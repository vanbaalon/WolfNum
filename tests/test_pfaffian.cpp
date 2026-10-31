// Pfaffians of real skew-symmetric matrices (Linalg::pfaffian; docs/numerics.md, "Pfaffian") against an independent MPFR replay
// of the documented blocked Parlett-Reid sequence, bit for bit. The reference keeps the full skew matrix with its original
// entries and applies every block's terms lazily when an entry is read (one reference::dot_sub per earlier block, then the
// in-block dot), exchanging whole rows and columns; the library applies earlier blocks eagerly on lower storage. GPU and host
// updates must agree, and Pf(A)^2 must agree with det(A) (factor_lu) to working accuracy.
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include <iostream>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int B> using F=Float<B>;
template<class T> bool same(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<int B> struct RefPf {F<B> value;std::size_t zero_pivot,status_column;int sign=1;};
template<int B> RefPf<B> reference_pfaffian(const std::vector<F<B>>& A,std::size_t n,std::size_t block){
    constexpr int N=B/32;RefPf<B> r;r.zero_pivot=r.status_column=n;
    if(n%2){word st=0;for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j)st|=A[i*n+j].status;r.value=zero<N>(st);return r;}
    r.value=zero<N>();r.value.limb[N-1]=0x80000000u;r.value.sign=1;if(!n)return r;
    const std::size_t nb=block?std::min(n,(block+1)/2*2):n;
    std::vector<F<B>> R(n*n,zero<N>());for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j){R[i*n+j]=A[i*n+j];R[j*n+i]=negate(A[i*n+j]);}
    std::vector<std::vector<F<B>>> P(n),Q(n);   // per index: [tau, y] / [y, -tau] of every step so far, blocks of nb
    std::vector<std::size_t> done;              // terms per finished block
    auto value=[&](std::size_t i,std::size_t c,std::size_t k0,std::size_t inblock){F<B> v=R[i*n+c];std::size_t off=0;
        for(std::size_t b=0;b*nb<k0;++b){v=reference::dot_sub<B>(v,P[i].data()+off,1,Q[c].data()+off,1,nb);off+=nb;}
        return reference::dot_sub<B>(v,P[i].data()+off,1,Q[c].data()+off,1,inblock);};
    for(auto& v:P)v.assign(n+nb,zero<N>());for(auto& v:Q)v.assign(n+nb,zero<N>());
    bool first=true;
    for(std::size_t k0=0;k0<n;k0+=nb)for(std::size_t k=k0,s=0;k<std::min(n,k0+nb);k+=2,++s){
        const std::size_t off=k0;   // P/Q offset of this block (nb terms per earlier block)
        std::vector<F<B>> x(n);word st=0;std::size_t p=k+1;
        for(std::size_t i=k+1;i<n;++i){x[i]=value(i,k,k0,2*s);st|=x[i].status;if(magnitude_compare(x[i],x[p])>0)p=i;}
        if(st){r.status_column=k;r.value=zero<N>(st);return r;}
        if(!x[p].sign){r.zero_pivot=k;r.value=zero<N>();return r;}
        if(p!=k+1){const std::size_t a=k+1;for(std::size_t c=0;c<n;++c)std::swap(R[a*n+c],R[p*n+c]);for(std::size_t i=0;i<n;++i)std::swap(R[i*n+a],R[i*n+p]);
            std::swap(P[a],P[p]);std::swap(Q[a],Q[p]);std::swap(x[a],x[p]);r.sign=-r.sign;}
        const F<B> a=negate(x[k+1]);r.value=first?a:reference::real<B>(Operation::mul,r.value,a);first=false;
        for(std::size_t i=k+2;i<n;++i){const F<B> y=value(i,k+1,k0,2*s),tau=reference::real<B>(Operation::div,x[i],a);
            P[i][off+2*s]=tau;P[i][off+2*s+1]=y;Q[i][off+2*s]=y;Q[i][off+2*s+1]=negate(tau);}
    }
    if(r.sign<0)r.value=negate(r.value);
    return r;
}
template<int B> std::vector<F<B>> skew(std::size_t n,std::mt19937_64& rng,int span){
    std::vector<F<B>> A(n*n);for(auto& x:A)x=reference::random_number<B>(rng,span); // upper part is arbitrary: never read
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j)if(rng()%11==0)A[i*n+j]=zero<B/32>();return A;}
template<int B> void check(Linalg& la,const std::vector<F<B>>& A,std::size_t n,std::size_t nb,const std::string& what){
    FactorOptions o;o.block=nb;o.host_macs=0;FactorOptions h=o;h.gpu=false;F<B> g,c;
    auto gi=la.pfaffian(B,A.data(),n,&g,o),hi=la.pfaffian(B,A.data(),n,&c,h);auto ref=reference_pfaffian<B>(A,n,nb);
    require(reference::equal<B>(g,ref.value),what+": Pfaffian != MPFR replay");require(same(g,c),what+": GPU and host differ");
    require(gi.zero_pivot==ref.zero_pivot&&gi.status_column==ref.status_column&&gi.sign==hi.sign,what+": info");
    const std::size_t blk=nb?(nb+1)/2*2:n;if(n%2==0&&n>blk&&ref.status_column==n&&ref.zero_pivot==n)require(gi.gpu_updates>0,what+": no GPU update");
    if(n%2==0&&n&&ref.status_column==n&&ref.zero_pivot==n){
        std::vector<F<B>> full(n*n,zero<B/32>());for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j){full[i*n+j]=A[i*n+j];full[j*n+i]=negate(A[i*n+j]);}
        F<B> det;la.factor_lu(B,full.data(),n).determinant(&det);F<B> sq=reference::real<B>(Operation::mul,g,g),diff=reference::real<B>(Operation::sub,sq,det);
        require(!diff.sign||diff.exponent<det.exponent-(B-40),what+": Pf^2 differs from det");}
}
template<int B> void width(){
    std::mt19937_64 rng(23+B);Linalg la;
    for(std::size_t n:{0,1,2,3,4,7,10,36,64,71})for(std::size_t nb:{0,2,3,8,16,32}){
        auto A=skew<B>(n,rng,n%3?5:200);check<B>(la,A,n,nb,"bits="+std::to_string(B)+" n="+std::to_string(n)+" nb="+std::to_string(nb));}
    {const std::size_t n=40;auto A=skew<B>(n,rng,5);A[33*n+20].status=exponent_overflow;check<B>(la,A,n,8,"status entry");
        auto Z=skew<B>(n,rng,5);for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<i;++j)if(i==12||j==12)Z[i*n+j]=zero<B/32>();check<B>(la,Z,n,8,"zero row and column");
        F<B> v;auto info=la.pfaffian(B,Z.data(),n,&v);require(!v.sign&&!v.status&&info.zero_pivot<n,"zero column: Pf != 0 or no zero pivot");}
    // 2x2 blocks: Pf = product of the superdiagonal entries exactly when it is representable.
    {const std::size_t n=8;std::vector<F<B>> A(n*n,zero<B/32>());F<B> want=zero<B/32>();
        for(std::size_t k=0;k<n;k+=2){F<B> v=zero<B/32>();v.limb[B/32-1]=0x80000000u;v.sign=k%4?-1:1;v.exponent=int(k);A[(k+1)*n+k]=negate(v);want=k?reference::real<B>(Operation::mul,want,v):v;}
        F<B> g;la.pfaffian(B,A.data(),n,&g);require(reference::equal<B>(g,want),"block-diagonal Pfaffian");}
    std::cout<<B<<" bits: Pfaffians match the MPFR replay, GPU and host agree, Pf^2 matches det\n";
}
int main(){try{width<256>();width<1152>();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
