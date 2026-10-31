// Batched block-diagonal congruence N = T M T^T (Engine::congruence, CommandBatch::congruence; docs/numerics.md, "Block-diagonal
// congruence") against an MPFR evaluation of the documented sequence (two cfma chains in ascending index, reference::complex_fused),
// bit for bit: block sizes 1, 3 and 4, complex and real-part output, full / lower / upper / strict triangles with untouched entries,
// leading dimensions, host and resident forms, and the rejected shapes.
#include "reference.hpp"
#include "limbforge/engine.hpp"
#include <cstring>
#include <iostream>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<class T> bool same(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<int B> void width(Engine& e){
    using F=Float<B>;using C=Complex<B/32>;std::mt19937_64 rng(B+5);
    auto rnd=[&]{return C{reference::random_number<B>(rng,6),reference::random_number<B>(rng,6)};};
    for(std::size_t b:{1,3,4})for(int tri:{0,1,2})for(bool strict:{false,true})for(bool real:{false,true}){
        if(strict&&!tri)continue;
        const std::size_t n=8*b,ldm=n+3,ldn=n+2;std::vector<C> T(n*b),M(n*ldm);for(auto& x:T)x=rnd();for(auto& x:M)x=rnd();
        Congruence s;s.n=n;s.block=b;s.ld_m=ldm;s.ld_n=ldn;s.real_output=real;s.triangle=tri;s.strict=strict;
        auto want=[&](std::size_t i,std::size_t j){const std::size_t P=i/b,Q=j/b;C acc{zero<B/32>(),zero<B/32>()};
            for(std::size_t l=0;l<b;++l){C t{zero<B/32>(),zero<B/32>()};for(std::size_t k=0;k<b;++k)t=reference::complex_fused<B>(T[(P*b+i%b)*b+k],M[(P*b+k)*ldm+Q*b+l],t);
                acc=reference::complex_fused<B>(t,T[(Q*b+j%b)*b+l],acc);}return acc;};
        auto written=[&](std::size_t i,std::size_t j){return !tri||(tri==1?(strict?i>j:i>=j):(strict?i<j:i<=j));};
        const std::string what="bits="+std::to_string(B)+" b="+std::to_string(b)+" tri="+std::to_string(tri)+" strict="+std::to_string(strict)+" real="+std::to_string(real);
        const std::size_t count=(n-1)*ldn+n;
        for(bool resident:{false,true}){
            if(real){std::vector<F> N0(count),N(count);for(auto& x:N0)x=reference::random_number<B>(rng,3);N=N0;
                if(!resident)e.congruence(B,s,T.data(),M.data(),N.data());
                else{auto Tb=e.make_buffer<C>(T.size()),Mb=e.make_buffer<C>(M.size());auto Nb=e.make_buffer<F>(count);Tb.upload(T.data(),T.size());Mb.upload(M.data(),M.size());Nb.upload(N0.data(),count);
                    auto batch=e.batch();batch.congruence(s,Tb,Mb,Nb);batch.submit().wait();Nb.download(N.data(),count);}
                for(std::size_t p=0;p<count;++p){const std::size_t i=p/ldn,j=p%ldn;
                    if(j<n&&written(i,j))require(reference::equal<B>(N[p],want(i,j).re),what+(resident?" resident":" host")+": entry "+std::to_string(p));
                    else require(same(N[p],N0[p]),what+": untouched entry "+std::to_string(p));}}
            else{std::vector<C> N0(count),N(count);for(auto& x:N0)x=rnd();N=N0;
                if(!resident)e.congruence(B,s,T.data(),M.data(),N.data());
                else{auto Tb=e.make_buffer<C>(T.size()),Mb=e.make_buffer<C>(M.size()),Nb=e.make_buffer<C>(count);Tb.upload(T.data(),T.size());Mb.upload(M.data(),M.size());Nb.upload(N0.data(),count);
                    auto batch=e.batch();batch.congruence(s,Tb,Mb,Nb);batch.submit().wait();Nb.download(N.data(),count);}
                for(std::size_t p=0;p<count;++p){const std::size_t i=p/ldn,j=p%ldn;
                    if(j<n&&written(i,j))require(reference::equal_complex<B>(N[p],want(i,j)),what+(resident?" resident":" host")+": entry "+std::to_string(p));
                    else require(same(N[p],N0[p]),what+": untouched entry "+std::to_string(p));}}}
    }
    bool threw=false;try{Congruence s;s.n=10;s.block=4;std::vector<C> x(400);e.congruence(B,s,x.data(),x.data(),x.data());}catch(const std::invalid_argument&){threw=true;}
    require(threw,"block not dividing n accepted");
    std::cout<<B<<" bits: block-diagonal congruences match MPFR (complex and real output, triangles, host and resident)\n";
}
int main(){try{Engine e;width<192>(e);width<256>(e);width<1152>(e);return 0;}catch(const std::exception& x){std::cerr<<x.what()<<'\n';return 1;}}
