#include "sc_pre.hpp"
#include "limbforge/mpfr_bridge.hpp"
#include <fstream>
#include <iostream>
using namespace sc;
void save(std::ofstream& out,const OrthoPoly& p){std::uint64_t size=p.b.size();out.write(reinterpret_cast<char*>(&size),8);limbforge::Complex<36> x;limbforge::from_mpc(1152,p.mu0.v,&x);out.write(reinterpret_cast<char*>(&x),sizeof(x));for(auto& v:p.b){limbforge::from_mpc(1152,v.v,&x);out.write(reinterpret_cast<char*>(&x),sizeof(x));}}
int main(){qsc::set_prec((mpfr_prec_t)std::ceil(320*3.3219281)+64);std::cout<<"g100, P640/Q540, WP270, input "<<qsc::prec_ref()<<" bits: building production measures"<<std::endl;
auto P=build_measures(C(200),642,0,295,2);std::cout<<"P measures converged at level "<<P.level<<std::endl;auto Q=build_measures(C(220),542,0,295,1);std::cout<<"Q measures converged at level "<<Q.level<<std::endl;
std::ofstream out("work/gpu-required/g100_basis.bin",std::ios::binary);save(out,P.P1);save(out,P.P2);save(out,Q.Q1);save(out,Q.Q2);if(!out)throw std::runtime_error("basis write failed");std::cout<<"Captured exact binary basis coefficients for both production GEMMs"<<std::endl;}
