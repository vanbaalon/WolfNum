#include "reference.hpp"
#include "limbforge/batched_linalg.hpp"
#include "limbforge/numerics.hpp"
#include "limbforge/capabilities.hpp"
#include <algorithm>
using namespace limbforge;
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
template<int B> void check(Engine& e,Linalg& la,BatchedLinalg& bl,Numerics& nm){
    using F=Float<B>;using C=Complex<B/32>;std::mt19937_64 rng(B);
    Prewarm warm;warm.bits={B};warm.reductions=true;e.prewarm(warm);
    constexpr std::size_t count=37;std::vector<F> a(count),b(count),out(count);
    for(std::size_t i=0;i<count;++i){a[i]=reference::random_number<B>(rng,4);b[i]=reference::random_number<B>(rng,4);}
    // Cancellation, subnormal-to-binary64 values, long carry chains, and every arithmetic pipeline used by QR.
    a[0]=b[0];a[1].exponent=b[1].exponent=-1400;a[4].exponent=3000;b[4].exponent=-3000;a[5].exponent=-3000;b[5].exponent=3000;
    for(auto op:{Operation::add,Operation::sub,Operation::mul,Operation::div,Operation::square,Operation::sqrt}){
        if(op==Operation::sqrt)for(auto& v:a)v.sign=1;
        if(operation_is_unary(op))e.run_unary(B,op,a.data(),out.data(),count);else e.run(B,op,a.data(),b.data(),out.data(),count);
        for(std::size_t i=0;i<count;++i)require(reference::equal<B>(out[i],reference::real<B>(op,a[i],b[i])),"wide arithmetic != MPFR");
    }
    for(auto op:{Operation::fma,Operation::fms}){
        e.run_ternary(B,op,a.data(),b.data(),a.data(),out.data(),count);
        for(std::size_t i=0;i<count;++i)require(reference::equal<B>(out[i],reference::fused<B>(a[i],b[i],a[i],op==Operation::fms)),"wide fused arithmetic != MPFR");
    }
    auto input=e.make_buffer<F>(count),sum=e.make_buffer<F>(1);input.upload(a.data(),count);
    {auto batch=e.batch();batch.tree_sum(input,sum);batch.submit().wait();}sum.download(out.data(),1);
    auto tree=a;while(tree.size()>1){std::vector<F> level((tree.size()+1)/2);for(std::size_t i=0;i<level.size();++i)
        level[i]=i*2+1<tree.size()?reference::real<B>(Operation::add,tree[2*i],tree[2*i+1]):tree[2*i];tree=std::move(level);}
    require(reference::equal<B>(out[0],tree[0]),"wide tree reduction != MPFR sequence");
    std::vector<C> ac(12),bc(12),oc(12),want(12);
    for(auto& x:ac)x={reference::random_number<B>(rng,4),reference::random_number<B>(rng,4)};
    for(auto& x:bc)x={reference::random_number<B>(rng,4),reference::random_number<B>(rng,4)};
    for(auto op:{Operation::complex_add,Operation::complex_mul,Operation::complex_div}){
        e.run(B,op,ac.data(),bc.data(),oc.data(),ac.size());
        for(std::size_t i=0;i<ac.size();++i)if(!reference::equal_complex<B>(oc[i],reference::complex<B>(op,ac[i],bc[i])))throw std::runtime_error(std::to_string(B)+" bits: complex arithmetic mismatch op="+std::to_string(int(op))+" entry="+std::to_string(i));
    }
    std::vector<Float<64>> narrow(count);e.cast(B,64,false,a.data(),narrow.data(),count);e.cast(64,B,false,narrow.data(),out.data(),count);
    reference::MP cast_ref(B);for(std::size_t i=0;i<count;++i){to_mpfr<B>(cast_ref.x,a[i]);auto low=from_mpfr<64>(cast_ref.x);
        require(reference::equal<64>(low,narrow[i]),"wide narrowing != MPFR");to_mpfr<64>(cast_ref.x,low);
        require(reference::equal<B>(out[i],from_mpfr<B>(cast_ref.x)),"wide widening != MPFR");}
    ac[0].re.exponent=3000;ac[1].im.exponent=-3000;bc[0].im.exponent=-1400;
    // Two batched 2x2 products, broadcasting a constant right matrix, including repeated resident submissions.
    StridedGemm s{2,2,2,3,6,0,4};auto ab=e.make_buffer<C>(12),bb=e.make_buffer<C>(6),cb=e.make_buffer<C>(8);
    ab.upload(ac.data(),12);bb.upload(bc.data(),6);
    for(bool fused:{false,true}){s.fused=fused;auto batch=e.batch();bl.gemm(batch,s,ab,bb,cb);batch.submit().wait();cb.download(oc.data(),8);
        for(std::size_t t=0;t<2;++t)for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j){C z{zero<B/32>(),zero<B/32>()};
            for(std::size_t k=0;k<3;++k){auto x=ac[t*6+i*3+k],y=bc[k*2+j];
                z=fused?reference::complex_fused<B>(x,y,z):reference::complex<B>(Operation::complex_add,z,reference::complex<B>(Operation::complex_mul,x,y));}
            if(!reference::equal_complex<B>(oc[t*4+i*2+j],z))throw std::runtime_error(std::to_string(B)+" bits: batched complex GEMM mismatch, fused="+std::to_string(fused)+", entry="+std::to_string(t*4+i*2+j)+", real equal="+std::to_string(reference::equal<B>(oc[t*4+i*2+j].re,z.re))+", imag equal="+std::to_string(reference::equal<B>(oc[t*4+i*2+j].im,z.im)));}
    }
    // Dense outputs and repeated cached-operand submissions catch compiler defects missed by tiny shapes.
    constexpr std::size_t batches=33;std::vector<C> da(batches*16),db(16),dc(batches*16),dw(batches*16);
    for(auto* v:{&da,&db})for(auto& x:*v)x={reference::random_number<B>(rng,4),reference::random_number<B>(rng,4)};
    auto dab=e.make_buffer<C>(da.size()),dbb=e.make_buffer<C>(db.size()),dcb=e.make_buffer<C>(dc.size());dab.upload(da.data(),da.size());dbb.upload(db.data(),db.size());
    StridedGemm dense{batches,4,4,4,16,0,16};for(bool fused:{false,true}){dense.fused=fused;
        for(std::size_t t=0;t<batches;++t)for(std::size_t i=0;i<4;++i)for(std::size_t j=0;j<4;++j){C z{zero<B/32>(),zero<B/32>()};
            for(std::size_t k=0;k<4;++k){auto x=da[t*16+i*4+k],y=db[k*4+j];z=fused?reference::complex_fused<B>(x,y,z):reference::complex<B>(Operation::complex_add,z,reference::complex<B>(Operation::complex_mul,x,y));}dw[t*16+i*4+j]=z;}
        for(int repeat=0;repeat<3;++repeat){auto batch=e.batch();bl.gemm(batch,dense,dab,dbb,dcb);batch.submit().wait();dcb.download(dc.data(),dc.size());
            for(std::size_t i=0;i<dc.size();++i)require(reference::equal_complex<B>(dc[i],dw[i]),"dense/repeated wide GEMM != MPFR");}}
    // Single-round exact real GEMM (also the QR trailing-update backend), including cancellation.
    la.gemm(B,false,a.data(),b.data(),2,3,5,out.data());
    for(int i=0;i<2;++i)for(int j=0;j<3;++j)require(reference::equal<B>(out[i*3+j],reference::dot<B>(a.data()+i*5,1,b.data()+j,3,5)),"wide exact GEMM != MPFR");
    // Exact norm_max must not underflow below binary64's range; tie breaks keep the first entry.
    a[0].exponent=-1400;a[1]=a[0];a[1].exponent=-1399;a[2]=a[1];a[3]=zero<B/32>();NormInfo info;
    nm.norm_max(B,Segments{1,4,false},a.data(),out.data(),&info);
    F maximum=a[1];maximum.sign=1;require(reference::equal<B>(out[0],maximum)&&info.index==1,"tiny norm_max underflow/tie failure");
    if constexpr(B==2048){
        // Cross several norm tree blocks at the largest width (key/index scratch must fit 32 KiB).
        F tiny=zero<B/32>();tiny.sign=1;tiny.exponent=-1400;tiny.limb[B/32-1]=0x80000000u;
        std::vector<F> many(257,tiny);many[129].exponent=-1399;
        nm.norm_max(B,Segments{1,many.size(),false},many.data(),out.data(),&info);
        require(reference::equal<B>(out[0],many[129])&&info.index==129,"wide multi-block norm_max failure");
        std::fill(many.begin(),many.end(),tiny);nm.norm2(B,Segments{1,many.size(),false},many.data(),out.data(),&info);
        reference::MP expected(B);mpfr_set_ui(expected.x,257,MPFR_RNDN);mpfr_sqrt(expected.x,expected.x,MPFR_RNDN);mpfr_mul_2si(expected.x,expected.x,-1400,MPFR_RNDN);
        require(reference::equal<B>(out[0],from_mpfr<B>(expected.x)),"wide multi-block norm2 != MPFR");
    }
    std::cout<<B<<" bits: arithmetic, exact GEMM, cached complex GEMM, tiny norms match MPFR"<<std::endl;
}
template<int B=1056> void widths(Engine& e,Linalg& la,BatchedLinalg& bl,Numerics& nm,int first=1056){if(B>=first)check<B>(e,la,bl,nm);if constexpr(B<2048)widths<B+32>(e,la,bl,nm,first);}
void capabilities(Engine& e){
    auto limits=e.device_limits();require(limits.max_buffer_bytes>0,"missing device limit");
    require(query_qr_support(limits,1152,3000,3000).supported,"3000x3000 real QR rejected");
    require(!query_qr_support(limits,2080,3,3).supported,"unsupported width accepted");
    require(!query_qr_support(limits,1152,2,3).supported,"wide QR shape accepted");
    require(!query_qr_support(limits,1152,SIZE_MAX,SIZE_MAX).supported,"overflow accepted");
    require(!query_qr_support(limits,1152,23171,23171,true).supported,"complex QR indexing overflow accepted");
    require(!query_gemm_support(limits,1152,3,3,65473).supported,"host-only inner width marked GPU eligible");
    DeviceLimits tiny{16,16,4096};require(!query_qr_support(tiny,1152,2,2).supported,"buffer limit ignored");
    StridedGemm s{2,2,2,3,6,0,4};require(query_batched_gemm_support(limits,1152,true,s).supported,"broadcast rejected");
    s.stride_c=3;require(!query_batched_gemm_support(limits,1152,true,s).supported,"overlapping outputs accepted");
}
void refinement(Linalg& la){
    // Condition number about 2^102, factor at 256 bits, residual/update at 1536 bits, target >1130 bits.
    reference::MP one(1536),epsilon(1536),perturb(1536),tmp(1536),error(1536),tol(1536);
    mpfr_set_ui(one.x,1,MPFR_RNDN);mpfr_set_ui_2exp(epsilon.x,1,-100,MPFR_RNDN);mpfr_set_ui_2exp(perturb.x,1,-500,MPFR_RNDN);
    reference::MP a00(1536),a01(1536),a10(1536),a11(1536),b0(1536),b1(1536),x0(1536),x1(1536),r0(1536),r1(1536);
    for(auto p:{a00.x,a01.x,a10.x,a11.x})mpfr_set_ui(p,1,MPFR_RNDN);mpfr_add(a11.x,a11.x,epsilon.x,MPFR_RNDN);
    mpfr_set_ui(b0.x,2,MPFR_RNDN);mpfr_add(b1.x,b0.x,epsilon.x,MPFR_RNDN);mpfr_add(b1.x,b1.x,perturb.x,MPFR_RNDN);
    std::vector<Float<256>> A={from_mpfr<256>(a00.x),from_mpfr<256>(a01.x),from_mpfr<256>(a10.x),from_mpfr<256>(a11.x)},rhs(2),delta(2);
    QROptions options;options.block=1;options.host_macs=options.solve_host_macs=0;auto f=la.factor_qr(256,A.data(),2,2,options);
    require(f.full_rank(),"refinement factor unexpectedly deficient");std::vector<unsigned char> original(4*sizeof(Float<256>));std::memcpy(original.data(),f.r(),original.size());
    int steps=0;for(;steps<12;++steps){
        mpfr_sub(r0.x,b0.x,x0.x,MPFR_RNDN);mpfr_sub(r0.x,r0.x,x1.x,MPFR_RNDN);
        mpfr_mul(tmp.x,a11.x,x1.x,MPFR_RNDN);mpfr_sub(r1.x,b1.x,x0.x,MPFR_RNDN);mpfr_sub(r1.x,r1.x,tmp.x,MPFR_RNDN);
        rhs[0]=from_mpfr<256>(r0.x);rhs[1]=from_mpfr<256>(r1.x);f.solve(rhs.data(),1,delta.data());
        to_mpfr<256>(tmp.x,delta[0]);mpfr_add(x0.x,x0.x,tmp.x,MPFR_RNDN);to_mpfr<256>(tmp.x,delta[1]);mpfr_add(x1.x,x1.x,tmp.x,MPFR_RNDN);
        mpfr_set_ui_2exp(tmp.x,1,-400,MPFR_RNDN);mpfr_add(tmp.x,one.x,tmp.x,MPFR_RNDN);mpfr_sub(error.x,x1.x,tmp.x,MPFR_RNDN);mpfr_abs(error.x,error.x,MPFR_RNDN);
        mpfr_set_ui_2exp(tol.x,1,-1130,MPFR_RNDN);if(mpfr_cmp(error.x,tol.x)<0)break;
    }
    require(steps<12,"mixed-precision refinement did not reach 1130 bits");
    require(std::memcmp(original.data(),f.r(),original.size())==0,"repeated solve changed the resident factor");
    std::cout<<"Reused 256-bit QR reached 1130-bit accuracy with full-precision residuals in "<<steps+1<<" solves"<<std::endl;
}
int main(int argc,char** argv){try{Engine e;Linalg la(e);BatchedLinalg bl(e);Numerics nm(e);std::cout<<e.device_name()<<std::endl;capabilities(e);refinement(la);
    if(argc==2&&std::string(argv[1])=="--all-widths")widths(e,la,bl,nm);
    else if(argc==2&&std::string(argv[1]).rfind("--wide-from=",0)==0){int first=std::stoi(std::string(argv[1]).substr(12));require(first>=1056&&first<=2048&&first%32==0,"invalid wide start");widths(e,la,bl,nm,first);}
    else {check<1056>(e,la,bl,nm);check<1152>(e,la,bl,nm);check<1536>(e,la,bl,nm);check<2048>(e,la,bl,nm);}
    return 0;}catch(const std::exception& ex){std::cerr<<ex.what()<<std::endl;return 1;}}
