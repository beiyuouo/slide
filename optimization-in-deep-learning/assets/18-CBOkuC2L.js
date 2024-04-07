import{_ as e}from"./slidev/KaTexBlockWrapper-BmT-VaEe.js";import{o as i,c as r,k as l,l as p,m as c,q as o,s as h,B as t,e as s,a1 as a}from"./modules/vue-Dim-dygt.js";import{I as g}from"./slidev/default-D-V2YAf6.js";import{b as u,U as n}from"./index-GjFdxGIJ.js";import{p as d,u as y,f as x}from"./slidev/context-B-08xyCf.js";import"./slidev/chunk-YNRNLEBD-B9teQT0a.js";import"./modules/shiki-D5uTpyAc.js";const v="/slide/optimization-in-deep-learning/strong-convex.png",w=s("h2",null,"Strong Convex",-1),f=s("br",null,null,-1),_=s("p",null,[a("通常我们会遇到凸函数，但是在实际中，我们更希望函数更加凸，即函数的曲率更大，这样可以加快函数的收敛速度。因此我们引入了 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"μ")]),s("annotation",{encoding:"application/x-tex"},"\\mu")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.625em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal"},"μ")])])]),a("-强凸的概念，"),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"μ")]),s("annotation",{encoding:"application/x-tex"},"\\mu")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.625em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal"},"μ")])])]),a("-强凸的定义如下：")],-1),b=s("p",null,[a("考虑实值函数 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"f"),s("mo",null,":"),s("msup",null,[s("mi",null,"R"),s("mi",null,"d")]),s("mo",null,"→"),s("mi",null,"R")]),s("annotation",{encoding:"application/x-tex"},"f: R^{d}\\to R")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8889em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},":"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8491em"}}),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.00773em"}},"R"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.8491em"}},[s("span",{style:{top:"-3.063em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mtight"},[s("span",{class:"mord mathnormal mtight"},"d")])])])])])])])]),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"→"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.6833em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.00773em"}},"R")])])]),a("，和模 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",{mathvariant:"normal"},"∣"),s("mi",{mathvariant:"normal"},"∣"),s("mo",null,"⋅"),s("mi",{mathvariant:"normal"},"∣"),s("mi",{mathvariant:"normal"},"∣")]),s("annotation",{encoding:"application/x-tex"},"||\\cdot||")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord"},"∣∣"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"⋅"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord"},"∣∣")])])]),a("，如果对任意自变量 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"x"),s("mo",{separator:"true"},","),s("mi",null,"y"),s("mo",null,"∈"),s("msup",null,[s("mi",null,"R"),s("mi",null,"d")])]),s("annotation",{encoding:"application/x-tex"},"x,y\\in R^d")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.7335em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"y"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"∈"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8491em"}}),s("span",{class:"mord"},[s("span",{class:"mord mathnormal",style:{"margin-right":"0.00773em"}},"R"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.8491em"}},[s("span",{style:{top:"-3.063em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mathnormal mtight"},"d")])])])])])])])])])]),a("，都有下面不等式成立：")],-1),k=s("p",null,[s("span",{class:"katex-display"},[s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("semantics",null,[s("mrow",null,[s("mi",null,"f"),s("mo",{stretchy:"false"},"("),s("mi",null,"y"),s("mo",{stretchy:"false"},")"),s("mo",null,"≥"),s("mi",null,"f"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("mo",{stretchy:"false"},")"),s("mo",null,"+"),s("mi",{mathvariant:"normal"},"∇"),s("mi",null,"f"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("msup",null,[s("mo",{stretchy:"false"},")"),s("mi",{mathvariant:"normal"},"⊤")]),s("mo",{stretchy:"false"},"("),s("mi",null,"y"),s("mo",null,"−"),s("mi",null,"x"),s("mo",{stretchy:"false"},")"),s("mo",null,"+"),s("mfrac",null,[s("mi",null,"μ"),s("mn",null,"2")]),s("mi",{mathvariant:"normal"},"∣"),s("mi",{mathvariant:"normal"},"∣"),s("mi",null,"y"),s("mo",null,"−"),s("mi",null,"x"),s("mi",{mathvariant:"normal"},"∣"),s("msup",null,[s("mi",{mathvariant:"normal"},"∣"),s("mn",null,"2")]),s("mo",{separator:"true"},","),s("mspace",{width:"1em"}),s("mi",{mathvariant:"normal"},"∀"),s("mi",null,"x"),s("mo",{separator:"true"},","),s("mi",null,"y"),s("mo",null,"∈"),s("mi",{mathvariant:"script"},"D")]),s("annotation",{encoding:"application/x-tex"},"f(y) \\geq f(x)+\\nabla f(x)^{\\top}(y-x)+\\frac{\\mu}{2}||y-x||^2, \\quad \\forall x, y \\in \\mathcal{D} ")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"y"),s("span",{class:"mclose"},")"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"≥"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mclose"},")"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"+"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1.1491em","vertical-align":"-0.25em"}}),s("span",{class:"mord"},"∇"),s("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f"),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mclose"},[s("span",{class:"mclose"},")"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.8991em"}},[s("span",{style:{top:"-3.113em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mtight"},[s("span",{class:"mord mtight"},"⊤")])])])])])])])]),s("span",{class:"mopen"},"("),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"y"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"−"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mclose"},")"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"+"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1.7936em","vertical-align":"-0.686em"}}),s("span",{class:"mord"},[s("span",{class:"mopen nulldelimiter"}),s("span",{class:"mfrac"},[s("span",{class:"vlist-t vlist-t2"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"1.1076em"}},[s("span",{style:{top:"-2.314em"}},[s("span",{class:"pstrut",style:{height:"3em"}}),s("span",{class:"mord"},[s("span",{class:"mord"},"2")])]),s("span",{style:{top:"-3.23em"}},[s("span",{class:"pstrut",style:{height:"3em"}}),s("span",{class:"frac-line",style:{"border-bottom-width":"0.04em"}})]),s("span",{style:{top:"-3.677em"}},[s("span",{class:"pstrut",style:{height:"3em"}}),s("span",{class:"mord"},[s("span",{class:"mord mathnormal"},"μ")])])]),s("span",{class:"vlist-s"},"​")]),s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.686em"}},[s("span")])])])]),s("span",{class:"mclose nulldelimiter"})]),s("span",{class:"mord"},"∣∣"),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"y"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"−"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1.1141em","vertical-align":"-0.25em"}}),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mord"},"∣"),s("span",{class:"mord"},[s("span",{class:"mord"},"∣"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.8641em"}},[s("span",{style:{top:"-3.113em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mtight"},"2")])])])])])])]),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"1em"}}),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mord"},"∀"),s("span",{class:"mord mathnormal"},"x"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"y"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"∈"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.6833em"}}),s("span",{class:"mord mathcal",style:{"margin-right":"0.02778em"}},"D")])])])])],-1),M=s("p",null,[a("则称函数 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"f")]),s("annotation",{encoding:"application/x-tex"},"f")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8889em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.10764em"}},"f")])])]),a(" 关于模 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",{mathvariant:"normal"},"∣"),s("mi",{mathvariant:"normal"},"∣"),s("mo",null,"⋅"),s("mi",{mathvariant:"normal"},"∣"),s("mi",{mathvariant:"normal"},"∣")]),s("annotation",{encoding:"application/x-tex"},"||\\cdot||")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord"},"∣∣"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"⋅"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mord"},"∣∣")])])]),a(" 是 "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"μ")]),s("annotation",{encoding:"application/x-tex"},"\\mu")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.625em","vertical-align":"-0.1944em"}}),s("span",{class:"mord mathnormal"},"μ")])])]),a("-强凸的。")],-1),z=s("div",{class:"flex flex-wrap justify-center gap-4"},[s("img",{src:v,class:"h-50 rounded"})],-1),L={__name:"18",setup(R){return d(n),y(),(B,P)=>{const m=e;return i(),r(g,o(h(t(x)(t(n),17))),{default:l(()=>[w,f,_,b,p(m,c({},{ranges:[]}),{default:l(()=>[k]),_:1},16),M,z]),_:1},16)}}},I=u(L,[["__file","/@slidev/slides/18.md"]]);export{I as default};
