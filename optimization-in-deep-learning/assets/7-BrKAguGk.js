import{o as n,c as e,k as m,q as o,s as i,B as t,e as a,a1 as s}from"./modules/vue-Dim-dygt.js";import{I as c}from"./slidev/default-D-V2YAf6.js";import{b as r,J as l}from"./index-GjFdxGIJ.js";import{p,u as h,f as x}from"./slidev/context-B-08xyCf.js";import"./modules/shiki-D5uTpyAc.js";const u="/slide/optimization-in-deep-learning/assets/output_optimization-intro_70d214_48_0.svg",d=a("h2",null,"Challenges in Deep Learning Optimization",-1),g=a("br",null,null,-1),w=a("h3",null,"1. 局部最小值(local minimum)",-1),f=a("br",null,null,-1),_=a("p",null,[s("对于任何目标函数 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"f"),a("mo",{stretchy:"false"},"("),a("mi",null,"x"),a("mo",{stretchy:"false"},")")]),a("annotation",{encoding:"application/x-tex"},"f(x)")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),a("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),a("span",{class:"mopen"},"("),a("span",{class:"mord mathnormal"},"x"),a("span",{class:"mclose"},")")])])]),s("，如果在 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"x")]),a("annotation",{encoding:"application/x-tex"},"x")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.4306em"}}),a("span",{class:"mord mathnormal"},"x")])])]),s(" 处对应的值 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"f"),a("mo",{stretchy:"false"},"("),a("mi",null,"x"),a("mo",{stretchy:"false"},")")]),a("annotation",{encoding:"application/x-tex"},"f(x)")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),a("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),a("span",{class:"mopen"},"("),a("span",{class:"mord mathnormal"},"x"),a("span",{class:"mclose"},")")])])]),s(" 小于在 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"x")]),a("annotation",{encoding:"application/x-tex"},"x")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.4306em"}}),a("span",{class:"mord mathnormal"},"x")])])]),s(" 附近任意其他点的 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"f"),a("mo",{stretchy:"false"},"("),a("mi",null,"x"),a("mo",{stretchy:"false"},")")]),a("annotation",{encoding:"application/x-tex"},"f(x)")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),a("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),a("span",{class:"mopen"},"("),a("span",{class:"mord mathnormal"},"x"),a("span",{class:"mclose"},")")])])]),s(" 值，那么 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"f"),a("mo",{stretchy:"false"},"("),a("mi",null,"x"),a("mo",{stretchy:"false"},")")]),a("annotation",{encoding:"application/x-tex"},"f(x)")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),a("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),a("span",{class:"mopen"},"("),a("span",{class:"mord mathnormal"},"x"),a("span",{class:"mclose"},")")])])]),s(" 可能是局部最小值。如果在 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"f"),a("mo",{stretchy:"false"},"("),a("mi",null,"x"),a("mo",{stretchy:"false"},")")]),a("annotation",{encoding:"application/x-tex"},"f(x)")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),a("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),a("span",{class:"mopen"},"("),a("span",{class:"mord mathnormal"},"x"),a("span",{class:"mclose"},")")])])]),s(" 处的 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"x")]),a("annotation",{encoding:"application/x-tex"},"x")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.4306em"}}),a("span",{class:"mord mathnormal"},"x")])])]),s(" 值是整个域中目标函数的最小值，那么 "),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mi",null,"f"),a("mo",{stretchy:"false"},"("),a("mi",null,"x"),a("mo",{stretchy:"false"},")")]),a("annotation",{encoding:"application/x-tex"},"f(x)")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),a("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),a("span",{class:"mopen"},"("),a("span",{class:"mord mathnormal"},"x"),a("span",{class:"mclose"},")")])])]),s(" 是全局最小值。")],-1),k=a("div",{class:"flex flex-wrap justify-center gap-4"},[a("img",{src:u,class:"h-60 rounded"})],-1),y={__name:"7",setup(M){return p(l),h(),(b,v)=>(n(),e(c,o(i(t(x)(t(l),6))),{default:m(()=>[d,g,w,f,_,k]),_:1},16))}},j=r(y,[["__file","/@slidev/slides/7.md"]]);export{j as default};