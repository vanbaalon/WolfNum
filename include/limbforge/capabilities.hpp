#pragma once
#include "batched_linalg.hpp"
#include <limits>
namespace limbforge {
// Structural preflight, without allocations or shader compilation. supported means the width, indexing and individual
// buffers fit the device. It does not promise free memory, a speedup, full rank, or absence of data-dependent fallback.
// Bytes describe matrix storage only; panel arithmetic, exponent-band/CRT workspaces and pipeline caches are extra.
struct ShapeSupport {
    bool supported=false;
    std::string reason;
    std::size_t operand_bytes=0,factor_bytes=0,workspace_matrix_bytes=0,largest_buffer_bytes=0;
};
namespace detail {
inline std::size_t support_mul(std::size_t a,std::size_t b){
    if(b&&a>std::numeric_limits<std::size_t>::max()/b)throw std::overflow_error("shape byte size overflow");return a*b;
}
inline void support_add(std::size_t& a,std::size_t b){
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::overflow_error("shape byte size overflow");a+=b;
}
inline std::size_t support_extent(std::size_t count,std::size_t stride,std::size_t matrix){
    if(!count||!matrix)return 0;auto n=support_mul(count-1,stride);support_add(n,matrix);return n;
}
inline void support_buffer(ShapeSupport& s,std::size_t& total,std::size_t bytes){support_add(total,bytes);s.largest_buffer_bytes=std::max(s.largest_buffer_bytes,bytes);}
inline ShapeSupport support_finish(ShapeSupport s,const DeviceLimits& limits){
    if(s.largest_buffer_bytes>limits.max_buffer_bytes){s.reason="matrix exceeds the device's maximum buffer length";return s;}
    s.supported=true;return s;
}
}
inline ShapeSupport query_batched_gemm_support(const DeviceLimits& limits,int bits,bool complex,const StridedGemm& g){
    ShapeSupport s;if(!precision_supported(bits)){s.reason="precision must be a multiple of 32 in [64,2048]";return s;}
    try {const auto a=detail::support_mul(g.m,g.k),b=detail::support_mul(g.k,g.n),c=detail::support_mul(g.m,g.n);
        if(detail::support_mul(g.count,c)>UINT32_MAX){s.reason="output grid exceeds 32-bit indexing";return s;}
        if(g.count>1&&g.stride_c<c){s.reason="output matrices overlap";return s;}
        auto element=std::size_t(bits/8+12)*(complex?2:1);
        for(auto pair:{std::make_pair(g.stride_a,a),std::make_pair(g.stride_b,b),std::make_pair(g.stride_c,c)})
            detail::support_buffer(s,s.operand_bytes,detail::support_mul(detail::support_extent(g.count,pair.first,pair.second),element));
        return detail::support_finish(s,limits);
    }catch(const std::overflow_error& e){s.reason=e.what();return s;}
}
inline ShapeSupport query_gemm_support(const DeviceLimits& limits,int bits,std::size_t m,std::size_t n,std::size_t k){
    ShapeSupport s;if(!precision_supported(bits)){s.reason="precision must be a multiple of 32 in [64,2048]";return s;}
    try {const auto grid=detail::support_mul(m,n);
        if(m>=(std::size_t(1)<<31)||n>=(std::size_t(1)<<31)||grid>=(std::size_t(1)<<31)){s.reason="matrix dimensions exceed 32-bit indexing";return s;}
        if(k>65472){s.reason="inner dimension exceeds the exact GPU backend limit (host fallback available)";return s;}
        const auto e=std::size_t(bits/8+12);
        for(auto count:{detail::support_mul(m,k),detail::support_mul(k,n),grid})detail::support_buffer(s,s.operand_bytes,detail::support_mul(count,e));
        return detail::support_finish(s,limits);
    }catch(const std::overflow_error& e){s.reason=e.what();return s;}
}
inline ShapeSupport query_qr_support(const DeviceLimits& limits,int bits,std::size_t m,std::size_t n,bool complex=false,std::size_t block=32){
    ShapeSupport s;if(!precision_supported(bits)){s.reason="precision must be a multiple of 32 in [64,2048]";return s;}
    if(m<n){s.reason="QR requires rows >= columns";return s;}
    try {const auto mn=detail::support_mul(m,n),nn=detail::support_mul(n,n);
        if(n>=(complex?std::size_t(23171):std::size_t(46341))||detail::support_mul(mn,complex?4:1)>=(std::size_t(1)<<31)){
            s.reason="QR matrix dimensions exceed 32-bit indexing";return s;}
        const auto e=std::size_t(bits/8+12),nb=block?std::min(block,n):n,blocks=nb?n/nb+(n%nb!=0):0;
        const auto tt=detail::support_mul(detail::support_mul(blocks,nb),nb);
        detail::support_buffer(s,s.operand_bytes,detail::support_mul(detail::support_mul(mn,complex?2:1),e));
        // Page-backed factors are rounded independently to the host page size.
        auto factor=[&](std::size_t count){auto bytes=detail::support_mul(count,e);const auto page=limits.host_page_bytes;
            if(page&&bytes){detail::support_add(bytes,page-1);bytes=bytes/page*page;}detail::support_buffer(s,s.factor_bytes,bytes);};
        for(auto count:{mn,nn,tt,n})factor(detail::support_mul(count,complex?2:1));
        if(complex){factor(detail::support_mul(mn,4));factor(detail::support_mul(tt,4));factor(detail::support_mul(nn,4));}
        if(n>nb){detail::support_buffer(s,s.workspace_matrix_bytes,detail::support_mul(detail::support_mul(mn,complex?2:1),e));
            for(int i=0;i<2;++i)detail::support_buffer(s,s.workspace_matrix_bytes,detail::support_mul(detail::support_mul(detail::support_mul(nb,n),complex?2:1),e));}
        return detail::support_finish(s,limits);
    }catch(const std::overflow_error& e){s.reason=e.what();return s;}
}
}
