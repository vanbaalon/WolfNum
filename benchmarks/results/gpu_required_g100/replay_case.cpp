#include "sc_pre.hpp"
#include "sc_gpu.hpp"
#include "reference.hpp"
#include <fstream>
#include <iostream>
using namespace sc;
C re(const C& z){C out;mpfr_set(mpc_realref(out.v),mpc_realref(z.v),MPFR_RNDN);return out;}
OrthoPoly load(std::ifstream& in){std::uint64_t count;in.read(reinterpret_cast<char*>(&count),8);if(!in||count>1000)throw std::runtime_error("bad basis count");OrthoPoly p;limbforge::Complex<36> v;in.read(reinterpret_cast<char*>(&v),sizeof(v));limbforge::to_mpc(1152,p.mu0.v,&v);p.b.resize(count);for(auto& x:p.b){in.read(reinterpret_cast<char*>(&v),sizeof(v));limbforge::to_mpc(1152,x.v,&v);}if(!in)throw std::runtime_error("bad basis data");return p;}
void default_routing(const Vec& a,const Vec& b,size_t m,size_t k,size_t n){
 using F=limbforge::Float<1152>;std::vector<F> ar(4*m*k),br(2*k*n),out(2*m*n);
 for(size_t i=0;i<m;++i)for(size_t q=0;q<k;++q){const C& z=a[i*k+q];F re=limbforge::from_mpfr<1152>(mpc_realref(z.v)),im=limbforge::from_mpfr<1152>(mpc_imagref(z.v));ar[(2*i)*2*k+2*q]=re;ar[(2*i)*2*k+2*q+1]=limbforge::negate(im);ar[(2*i+1)*2*k+2*q]=im;ar[(2*i+1)*2*k+2*q+1]=re;}
 for(size_t q=0;q<k;++q)for(size_t j=0;j<n;++j){br[(2*q)*n+j]=limbforge::from_mpfr<1152>(mpc_realref(b[q*n+j].v));br[(2*q+1)*n+j]=limbforge::from_mpfr<1152>(mpc_imagref(b[q*n+j].v));}
 limbforge::Linalg narrow;try{narrow.gemm_gpu(1152,false,ar.data(),br.data(),2*m,n,2*k,out.data());std::cout<<"Default options support these rows on GPU"<<std::endl;}catch(const std::runtime_error& e){std::cout<<"Default routing: "<<e.what()<<std::endl;}
}
void verify(const Vec& a,const Vec& b,const Vec& c,size_t m,size_t k,size_t n){
    using F=limbforge::Float<1152>;std::vector<F> br(2*k*n);for(size_t q=0;q<k;++q)for(size_t j=0;j<n;++j){br[(2*q)*n+j]=limbforge::from_mpfr<1152>(mpc_realref(b[q*n+j].v));br[(2*q+1)*n+j]=limbforge::from_mpfr<1152>(mpc_imagref(b[q*n+j].v));}
    std::vector<F> row(2*k);reference::MP expected(qsc::prec_ref());size_t checks=0;
    for(size_t i:{size_t(0),size_t(127),size_t(128),m/2,m-1})for(size_t j:{size_t(0),size_t(1),n/2,n-1})for(int part:{0,1}){
      for(size_t q=0;q<k;++q){const C& z=a[i*k+q];auto re=limbforge::from_mpfr<1152>(mpc_realref(z.v)),im=limbforge::from_mpfr<1152>(mpc_imagref(z.v));row[2*q]=part?im:re;row[2*q+1]=part?re:limbforge::negate(im);}
      F result=reference::dot<1152>(row.data(),1,br.data()+j,n,2*k);limbforge::to_mpfr<1152>(expected.x,result);
      if(mpfr_cmp(expected.x,part?mpc_imagref(c[i*n+j].v):mpc_realref(c[i*n+j].v))!=0)throw std::runtime_error("production MPFR sample mismatch at "+std::to_string(i)+","+std::to_string(j));++checks;
    }std::cout<<checks<<" real-component samples match single-round exact MPFR (including tile boundaries and last row)"<<std::endl;
}
int main(int argc,char** argv){try{qsc::set_prec((mpfr_prec_t)std::ceil(320*3.3219281)+64);std::ifstream in("work/gpu-required/g100_basis.bin",std::ios::binary);if(!in)throw std::runtime_error("basis capture not ready");auto P1=load(in),P2=load(in),Q1=load(in),Q2=load(in);
const int nP=640,nQ=540;Vec phi,wphi,psi,wpsi,rP,wrP,rQ,wrQ;gauss_rule(P1,nP,phi,wphi);gauss_rule(Q1,nQ,psi,wpsi);gauss_rule(P2,nP+1,rP,wrP);gauss_rule(Q2,nQ+1,rQ,wrQ);
C g(100),I=C::imag_unit(),two(2),tpiI=two*cpi()*I;std::vector<std::pair<double,C>> po;for(auto& r:rQ)if(qsc::real_d(r)>0){C x=qsc::exp(r),u=re(g*(x+C(1)/x));po.push_back({qsc::real_d(u),u});}for(auto& r:rP)if(qsc::real_d(r)>=0){C x=qsc::exp(I*r),u=re(g*(x+C(1)/x));po.push_back({qsc::real_d(u),u});}std::sort(po.begin(),po.end(),[](auto& a,auto& b){return a.first<b.first;});if(po.size()!=591)throw std::runtime_error("wrong point count");
Vec ys(5*po.size());for(int level=0;level<5;++level)for(size_t j=0;j<po.size();++j){C u=po[j].second+I*C(level);ys[level*po.size()+j]=(u+qsc::sqrt(u-two*g)*qsc::sqrt(u+two*g))/(two*g);}
const size_t NN=ys.size(),m=4*NN;const int threads=8;const bool small=argc>1&&std::string(argv[1])=="--sample-rows",routing=argc>1&&std::string(argv[1])=="--default-routing";
for(bool q:{false,true}){size_t k=q?nQ:nP,n=k+1,rows=routing?3:small?259:m;auto& nodes=q?psi:phi;auto& weights=q?wpsi:wphi;auto& op=q?Q2:P2;Vec b(k*n),xs(k),a(rows*k),c;
  par_for(int(k),threads,[&](int j){Vec p=ortho_eval(op,nodes[j],int(k));if(!q)for(size_t t=1;t<=k;t+=2)p[t]=p[t]*I;xs[j]=q?qsc::exp(nodes[j]):qsc::exp(I*nodes[j]);for(size_t t=0;t<n;++t)b[j*n+t]=(q?weights[j]*p[t]:-(weights[j]*p[t]))/tpiI;});
  par_for(int(rows),threads,[&](int rr){size_t source=small?(rr==258?m-1:size_t(rr)):size_t(rr);int block=int(source/NN);C y=ys[source%NN];if(block%2)y=C(1)/y;for(size_t j=0;j<k;++j)a[size_t(rr)*k+j]=(q?xs[j]:I*xs[j])/(block<2?xs[j]+y:xs[j]-y);});
  std::cout<<(q?"Q":"P")<<" production operands: "<<rows<<" x "<<k<<" x "<<n<<", measured spread "<<gpu::dense_spread(a,rows,k,b,n)<<" bits"<<std::endl;
  if(routing){default_routing(a,b,rows,k,n);continue;}
  if(!gpu::cgemm_t<1152>(a,rows,k,b,n,c,threads))throw std::runtime_error("case used outer CPU fallback");verify(a,b,c,rows,k,n);
}
std::cout<<(routing?"Default routing analysis completed":"Production reconstruction GEMM GPU-placement verification passed")<<std::endl;return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
