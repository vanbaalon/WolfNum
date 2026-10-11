#include "limbforge/capabilities.hpp"
#include <iostream>
using namespace limbforge;
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(){try{
    DeviceLimits limits{SIZE_MAX,SIZE_MAX,4096};
    for(int b=64;b<=2048;b+=32){require(precision_supported(b),"valid width rejected");
        auto q=query_qr_support(limits,b,3000,3000);require(q.supported,"valid square QR rejected");
        const std::size_t e=b/8+12,nb=32,t=94*nb*nb;std::size_t want=0;
        for(auto count:{std::size_t(9000000),std::size_t(9000000),t,std::size_t(3000)})want+=(count*e+4095)/4096*4096;
        require(q.factor_bytes==want,"factor estimate differs from allocated page blocks");
        require(q.workspace_matrix_bytes==(9000000+2*32*3000)*e,"real matrix scratch estimate wrong");}
    for(int b:{-64,0,32,65,2080})require(!precision_supported(b)&&!query_qr_support(limits,b,1,1).supported,"invalid precision accepted");
    require(!query_qr_support(limits,2048,46341,46341).supported,"real QR indexing bound ignored");
    require(query_qr_support(limits,2048,46340,46340).supported,"real QR indexing edge rejected");
    require(query_qr_support(limits,2048,23170,23170,true).supported,"complex QR indexing edge rejected");
    require(!query_qr_support(limits,2048,23171,23171,true).supported,"complex QR indexing bound ignored");
    require(!query_qr_support(limits,2048,SIZE_MAX,2).supported,"QR multiplication overflow accepted");
    StridedGemm g{2,2,3,4,0,12,6};auto s=query_batched_gemm_support(limits,2048,true,g);
    require(s.supported&&s.operand_bytes==(8+24+12)*536,"strided/broadcast storage estimate wrong");
    g.stride_a=SIZE_MAX;require(!query_batched_gemm_support(limits,2048,true,g).supported,"stride overflow accepted");
    g={2,2,3,4,0,12,5};require(!query_batched_gemm_support(limits,2048,true,g).supported,"overlapping outputs accepted");
    g={SIZE_MAX,2,3,4,0,0,6};require(!query_batched_gemm_support(limits,2048,true,g).supported,"grid overflow accepted");
    DeviceLimits tiny{4095,4095,4096};require(!query_qr_support(tiny,64,1,1).supported,"page-rounded buffer bound ignored");
    std::cout<<"Capability width, indexing, page storage, buffer and overflow boundaries pass\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
