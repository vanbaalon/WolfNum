#pragma once
// MPFR element policies for the LU and Pfaffian replays (tests/test_lu.cpp, tests/test_pfaffian.cpp): the documented complex
// division, exact |z|^2 comparison and one-rounding complex products and dots, computed independently with MPFR.
#include "reference.hpp"
#include <cstring>
#include <vector>
template<class T> bool same(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<int B> struct RR {
    using T=limbforge::Float<B>;static constexpr bool complex=false;
    static T rnd(std::mt19937_64& g,int span){return reference::random_number<B>(g,span);}
    static T zeros(limbforge::word s=0){return limbforge::zero<B/32>(s);}static T one(){T x=limbforge::zero<B/32>();x.limb[B/32-1]=0x80000000u;x.sign=1;return x;}
    static limbforge::word status(const T& x){return x.status;}static bool is_zero(const T& x){return !x.sign;}
    static int compare(const T& a,const T& b){T v[2]={a,b},w[2]={a,limbforge::negate(b)};return reference::dot<B>(v,1,w,1,2).sign;}
    static T quotient(const T& a,const T& b){return reference::real<B>(limbforge::Operation::div,a,b);}
    static T product(const T& a,const T& b){return reference::real<B>(limbforge::Operation::mul,a,b);}
    static T neg(const T& x){return limbforge::negate(x);}
    // c - sum_q x_q y_q with one rounding.
    static T D(const T& c,const std::vector<T>& x,const std::vector<T>& y){return reference::dot_sub<B>(c,x.data(),1,y.data(),1,x.size());}
    static bool equal(const T& a,const T& b){return reference::equal<B>(a,b);}
};
template<int B> struct RC {
    using F=limbforge::Float<B>;using T=limbforge::Complex<B/32>;static constexpr bool complex=true;
    static T rnd(std::mt19937_64& g,int span){return {reference::random_number<B>(g,span),reference::random_number<B>(g,span)};}
    static T zeros(limbforge::word s=0){return {limbforge::zero<B/32>(s),limbforge::zero<B/32>(s)};}static T one(){return {RR<B>::one(),limbforge::zero<B/32>()};}
    static limbforge::word status(const T& x){return x.re.status|x.im.status;}static bool is_zero(const T& x){return !x.re.sign&&!x.im.sign;}
    static F dot(std::vector<F> x,std::vector<F> y){return reference::dot<B>(x.data(),1,y.data(),1,x.size());}
    static int compare(const T& a,const T& b){return dot({a.re,a.im,b.re,b.im},{a.re,a.im,limbforge::negate(b.re),limbforge::negate(b.im)}).sign;}
    static F shift(F x,long e){if(x.sign)x.exponent=int(x.exponent+e);return x;}
    static T quotient(const T& s,const T& u){if(status(s)|status(u))return zeros(status(s)|status(u));if(is_zero(u))return zeros(limbforge::division_by_zero);
        long e=LONG_MIN;if(u.re.sign)e=u.re.exponent;if(u.im.sign)e=std::max(e,long(u.im.exponent));
        const F ur=shift(u.re,-e),ui=shift(u.im,-e),d=dot({ur,ui},{ur,ui});
        return {shift(reference::real<B>(limbforge::Operation::div,dot({s.re,s.im},{ur,ui}),d),-e),shift(reference::real<B>(limbforge::Operation::div,dot({s.im,s.re},{ur,limbforge::negate(ui)}),d),-e)};}
    static T product(const T& a,const T& b){return {dot({a.re,a.im},{b.re,limbforge::negate(b.im)}),dot({a.re,a.im},{b.im,b.re})};}
    static T neg(const T& x){return {limbforge::negate(x.re),limbforge::negate(x.im)};}
    static T D(const T& c,const std::vector<T>& x,const std::vector<T>& y){std::vector<F> l,r,i;
        for(std::size_t q=0;q<x.size();++q){l.push_back(x[q].re);l.push_back(x[q].im);r.push_back(y[q].re);r.push_back(limbforge::negate(y[q].im));i.push_back(y[q].im);i.push_back(y[q].re);}
        return {reference::dot_sub<B>(c.re,l.data(),1,r.data(),1,l.size()),reference::dot_sub<B>(c.im,l.data(),1,i.data(),1,l.size())};}
    static bool equal(const T& a,const T& b){return reference::equal<B>(a.re,b.re)&&reference::equal<B>(a.im,b.im);}
};
