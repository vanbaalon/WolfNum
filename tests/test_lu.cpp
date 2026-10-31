// Blocked LU with partial pivoting (Linalg::factor_lu, LUFactor::solve / determinant) against an independent MPFR replay of the
// documented rounding sequence (docs/numerics.md, "LU factorization"), bit for bit. The reference is left-looking, column by
// column: column j receives the U rows and trailing update of each earlier block (reference::dot_sub), then its in-block dots,
// the pivot search and the divisions (MPFR); whole rows are swapped as the pivots are chosen. GPU and host updates must agree.
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include <iostream>
#include <numeric>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int B> using F=Float<B>;
template<class T> bool same(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<int B> F<B> rdiv(const F<B>& a,const F<B>& b){return reference::real<B>(Operation::div,a,b);}
template<int B> F<B> rmul(const F<B>& a,const F<B>& b){return reference::real<B>(Operation::mul,a,b);}
template<int B> struct RefLU {std::vector<F<B>> W;std::vector<std::size_t> perm;int sign=1;std::size_t zero_pivot,status_column;};
template<int B> RefLU<B> reference_lu(std::vector<F<B>> W,std::size_t n,std::size_t nb){
    RefLU<B> r;r.perm.resize(n);std::iota(r.perm.begin(),r.perm.end(),0);r.zero_pivot=r.status_column=n;
    auto at=[&](std::size_t i,std::size_t j)->F<B>&{return W[i*n+j];};
    auto D=[&](F<B> c,std::size_t i,std::size_t j,std::size_t q0,std::size_t q1){return reference::dot_sub<B>(c,&W[i*n+q0],1,&W[q0*n+j],std::ptrdiff_t(n),q1-q0);};
    for(std::size_t j=0;j<n;++j){const std::size_t k0=j/nb*nb;
        for(std::size_t b0=0;b0<k0;b0+=nb){const std::size_t b1=b0+nb;                     // earlier blocks: U rows, then the trailing update
            for(std::size_t i=b0;i<b1;++i)at(i,j)=D(at(i,j),i,j,b0,i);
            for(std::size_t i=b1;i<n;++i)at(i,j)=D(at(i,j),i,j,b0,b1);}
        for(std::size_t i=k0;i<j;++i)at(i,j)=D(at(i,j),i,j,k0,i);
        for(std::size_t i=j;i<n;++i)at(i,j)=D(at(i,j),i,j,k0,j);
        word st=0;std::size_t p=j;for(std::size_t i=j;i<n;++i){st|=at(i,j).status;if(magnitude_compare(at(i,j),at(p,j))>0)p=i;}
        if(st){r.status_column=j;for(std::size_t i=0;i<n;++i)for(std::size_t c=j;c<n;++c)at(i,c)=zero<B/32>(invalid);break;}
        if(p!=j){for(std::size_t c=0;c<n;++c)std::swap(at(p,c),at(j,c));std::swap(r.perm[p],r.perm[j]);r.sign=-r.sign;}
        if(!at(j,j).sign){if(r.zero_pivot==n)r.zero_pivot=j;continue;}
        for(std::size_t i=j+1;i<n;++i)at(i,j)=rdiv<B>(at(i,j),at(j,j));
    }
    r.W=std::move(W);return r;
}
// X = P B; unit-lower forward and upper backward substitution, blocked as trsm (one dot_sub per block, then the in-block dot).
template<int B> std::vector<F<B>> reference_solve(const RefLU<B>& f,const std::vector<F<B>>& Bm,std::size_t n,std::size_t nrhs,std::size_t nb){
    std::vector<F<B>> X(n*nrhs);const auto& W=f.W;
    for(std::size_t i=0;i<n;++i)for(std::size_t c=0;c<nrhs;++c)X[i*nrhs+c]=Bm[f.perm[i]*nrhs+c];
    for(std::size_t c=0;c<nrhs;++c){auto x=[&](std::size_t i)->F<B>&{return X[i*nrhs+c];};
        for(std::size_t i=0;i<n;++i){const std::size_t r0=i/nb*nb;
            for(std::size_t b0=0;b0<r0;b0+=nb)x(i)=reference::dot_sub<B>(x(i),&W[i*n+b0],1,&X[b0*nrhs+c],std::ptrdiff_t(nrhs),nb);
            x(i)=reference::dot_sub<B>(x(i),&W[i*n+r0],1,&X[r0*nrhs+c],std::ptrdiff_t(nrhs),i-r0);}
        for(std::size_t i=n;i-->0;){const std::size_t r0=i/nb*nb,r1=std::min(n,r0+nb);
            for(std::size_t b0=(n-1)/nb*nb;b0>r0;b0-=nb){const std::size_t b1=std::min(n,b0+nb);
                x(i)=reference::dot_sub<B>(x(i),&W[i*n+b0],1,&X[b0*nrhs+c],std::ptrdiff_t(nrhs),b1-b0);}
            if(i+1<r1)x(i)=reference::dot_sub<B>(x(i),&W[i*n+i+1],1,&X[(i+1)*nrhs+c],std::ptrdiff_t(nrhs),r1-i-1);
            x(i)=rdiv<B>(x(i),W[i*n+i]);}}
    return X;
}
template<int B> std::vector<F<B>> random_matrix(std::size_t rows,std::size_t cols,std::mt19937_64& rng,int span){
    std::vector<F<B>> A(rows*cols);for(auto& x:A){x=reference::random_number<B>(rng,span);if(rng()%13==0)x=zero<B/32>();}return A;}
template<int B> void check(Linalg& gpu,Linalg& host,const std::vector<F<B>>& A,std::size_t n,std::size_t nb,const std::string& what,std::mt19937_64& rng){
    FactorOptions o;o.block=nb;o.host_macs=0;o.solve_host_macs=0;FactorOptions h=o;h.gpu=false;
    const std::size_t blk=nb?std::min(nb,std::max<std::size_t>(n,1)):std::max<std::size_t>(n,1);
    auto ref=reference_lu<B>(A,n,blk);auto f=gpu.factor_lu(B,A.data(),n,o),g=host.factor_lu(B,A.data(),n,h);
    const F<B>* W=static_cast<const F<B>*>(f.lu());const F<B>* V=static_cast<const F<B>*>(g.lu());
    for(std::size_t p=0;p<n*n;++p){require(reference::equal<B>(W[p],ref.W[p]),what+": factor entry "+std::to_string(p)+" != MPFR replay");require(same(W[p],V[p]),what+": GPU and host factors differ");}
    for(std::size_t i=0;i<n;++i)require(f.permutation()[i]==ref.perm[i],what+": permutation");
    require(f.permutation_sign()==ref.sign&&f.info().zero_pivot==ref.zero_pivot&&f.info().status_column==ref.status_column,what+": info");
    if(n>nb&&nb&&ref.status_column==n)require(f.info().gpu_updates>0,what+": no GPU update");
    const std::size_t nrhs=3;auto Bm=random_matrix<B>(n,nrhs,rng,4);std::vector<F<B>> X(n*nrhs),Y(n*nrhs);f.solve(Bm.data(),nrhs,X.data());g.solve(Bm.data(),nrhs,Y.data());
    if(ref.status_column==n){auto want=reference_solve<B>(ref,Bm,n,nrhs,blk);
        for(std::size_t p=0;p<X.size();++p){require(reference::equal<B>(X[p],want[p]),what+": solve entry "+std::to_string(p)+" != MPFR replay");require(same(X[p],Y[p]),what+": GPU and host solves differ");}}
    else for(auto& x:X)require(x.status==invalid&&!x.sign,what+": solve of a status factor");
    F<B> det;f.determinant(&det);F<B> want=zero<B/32>(invalid);
    if(ref.status_column==n){if(!n){want=zero<B/32>();want.limb[B/32-1]=0x80000000u;want.sign=1;}else{want=ref.W[0];for(std::size_t i=1;i<n;++i)want=rmul<B>(want,ref.W[i*n+i]);if(ref.sign<0)want=negate(want);}}
    require(reference::equal<B>(det,want),what+": determinant");
}
template<int B> void width(){
    std::mt19937_64 rng(11+B);Linalg gpu,host;
    for(std::size_t n:{0,1,2,5,37,70})for(std::size_t nb:{0,1,8,16,32}){
        auto A=random_matrix<B>(n,n,rng,n%2?6:300);check<B>(gpu,host,A,n,nb,"bits="+std::to_string(B)+" n="+std::to_string(n)+" nb="+std::to_string(nb),rng);}
    // Duplicate rows (rounding usually leaves a tiny pivot), a status entry (stops) and a zero column (an exact zero pivot; LAPACK continues).
    {const std::size_t n=40;auto A=random_matrix<B>(n,n,rng,6);for(std::size_t c=0;c<n;++c)A[17*n+c]=A[3*n+c];check<B>(gpu,host,A,n,16,"equal rows",rng);
        auto S=random_matrix<B>(n,n,rng,6);S[25*n+30].status=exponent_overflow;check<B>(gpu,host,S,n,16,"status entry",rng);
        auto Z=random_matrix<B>(n,n,rng,6);for(std::size_t r=0;r<n;++r)Z[r*n+9]=zero<B/32>();check<B>(gpu,host,Z,n,8,"zero column",rng);
        auto f=gpu.factor_lu(B,Z.data(),n);require(!f.nonsingular()&&f.info().zero_pivot==9,"zero column not reported as zero pivot 9");}
    std::cout<<B<<" bits: LU factors, permutations, solves and determinants match the MPFR replay; GPU and host updates agree\n";
}
int main(){try{width<256>();width<1152>();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
