#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include <iostream>
using namespace limbforge;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
template<int B> void width(){
    using F=Float<B>;constexpr int N=B/32;std::mt19937_64 rng(B);constexpr std::size_t m=3,n=4,k=7;
    std::vector<F> a(m*k),b(k*n),out(m*n),sentinel(m*n);
    for(auto& x:a)x=reference::random_number<B>(rng,4);
    for(std::size_t i=0;i<k;++i)for(std::size_t j=0;j<n;++j){b[i*n+j]=reference::random_number<B>(rng,4);b[i*n+j].exponent-=int(i)*150;}
    for(auto& x:sentinel)x=reference::random_number<B>(rng,4);out=sentinel;
    Linalg narrow;bool threw=false;try{narrow.gemm_gpu(B,false,a.data(),b.data(),m,n,k,out.data());}catch(const std::runtime_error&){threw=true;}
    require(threw&&narrow.report().right_fallback==n,"GPU requirement did not reject host products");
    require(!std::memcmp(out.data(),sentinel.data(),out.size()*sizeof(F)),"fallback rejection modified C");
    LinalgOptions options;options.band_bits=512;options.max_bands=3;options.max_spread=1024;Linalg wide(options);
    auto t=wide.gemm_gpu(B,false,a.data(),b.data(),m,n,k,out.data());
    require(wide.report().gpu_outputs==m*n&&!wide.report().fallback_outputs&&t.gpu_seconds>0,"strict product was not entirely GPU");
    for(std::size_t i=0;i<m;++i)for(std::size_t j=0;j<n;++j)require(reference::equal<B>(out[i*n+j],reference::dot<B>(a.data()+i*k,1,b.data()+j,n,k)),"GPU-required GEMM != MPFR");
    out=sentinel;wide.gemm_gpu(B,false,a.data(),b.data(),m,n,k,out.data(),true);
    for(std::size_t i=0;i<m;++i)for(std::size_t j=0;j<n;++j)require(reference::equal<B>(out[i*n+j],reference::dot_sub<B>(sentinel[i*n+j],a.data()+i*k,1,b.data()+j,n,k)),"GPU-required update != MPFR");
    std::vector<F> at(k*m);for(std::size_t i=0;i<m;++i)for(std::size_t q=0;q<k;++q)at[q*m+i]=a[i*k+q];
    wide.gemm_gpu(B,true,at.data(),b.data(),m,n,k,out.data());
    for(std::size_t i=0;i<m;++i)for(std::size_t j=0;j<n;++j)require(reference::equal<B>(out[i*n+j],reference::dot<B>(a.data()+i*k,1,b.data()+j,n,k)),"GPU-required transpose != MPFR");
    // Zero/status products need no device dot, even with a would-be fallback on the other side.
    std::fill(a.begin(),a.end(),zero<N>());a[0].status=invalid;narrow.gemm_gpu(B,false,a.data(),b.data(),m,n,k,out.data());
    require(narrow.report().trivial_outputs==m*n&&!narrow.report().fallback_outputs,"trivial results falsely require device work");
    for(std::size_t i=0;i<m;++i)for(std::size_t j=0;j<n;++j)require(reference::equal<B>(out[i*n+j],zero<N>(i==0?invalid:0)),"trivial status contract changed");
    LinalgOptions forced;forced.force_fallback=true;Linalg host(forced);a[0]=reference::random_number<B>(rng,4);out=sentinel;threw=false;
    try{host.gemm_gpu(B,false,a.data(),b.data(),m,n,k,out.data());}catch(const std::runtime_error&){threw=true;}
    require(threw&&!std::memcmp(out.data(),sentinel.data(),out.size()*sizeof(F)),"forced host path bypassed strict mode");
    std::cout<<B<<" bits: strict GPU products/updates/transpose match MPFR; fallback rejected before writing C\n";
}
int main(){try{width<512>();width<928>();width<1024>();width<1152>();width<1536>();width<2048>();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
