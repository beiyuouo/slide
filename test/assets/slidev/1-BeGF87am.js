import{i as x,w as b,x as g}from"../modules/unplugin-icons-CXaKIwH_.js";import{d as k,y as $,o as _,b as y,e as t,f as w,h as C,c as P,k as S,A as r,a0 as B,l as c,q as E,s as z}from"../modules/vue-CbUuoOfz.js";import{u,p as I,f as N}from"./context-Cki04sHT.js";import{b as m,E as d}from"../index-Da20ahjm.js";import"../modules/shiki-B43ebrIS.js";function p(e){return e.startsWith("/")?"/slide/test"+e.slice(1):e}function V(e,o=!1){const s=e&&["#","rgb","hsl"].some(i=>e.indexOf(i)===0),n={background:s?e:void 0,color:e&&!s?"white":void 0,backgroundImage:s?void 0:e?o?`linear-gradient(#0005, #0008), url(${p(e)})`:`url("${p(e)}")`:void 0,backgroundRepeat:"no-repeat",backgroundPosition:"center",backgroundSize:"cover"};return n.background||delete n.background,n}const j={class:"my-auto w-full"},A=k({__name:"cover",props:{background:{default:"https://source.unsplash.com/collection/94734566/1920x1080"}},setup(e){u();const o=e,s=$(()=>V(o.background,!0));return(n,i)=>(_(),y("div",{class:"slidev-layout cover text-center",style:C(s.value)},[t("div",j,[w(n.$slots,"default")])],4))}}),O=m(A,[["__file","/home/runner/work/slide/slide/test/node_modules/@slidev/theme-seriph/layouts/cover.vue"]]),R=t("h1",null,"Welcome to Slidev",-1),T=t("p",null,"Presentation slides for developers",-1),W={class:"pt-12"},q={class:"abs-br m-6 flex gap-2"},F={href:"https://github.com/slidevjs/slidev",target:"_blank",alt:"GitHub",class:"text-xl icon-btn opacity-50 !border-none !hover:text-white"},G={__name:"1",setup(e){I(d);const{$slidev:o,$nav:s,$clicksContext:n,$clicks:i,$page:H,$renderContext:L,$frontmatter:U}=u();return(D,a)=>{const v=x,f=b,h=g;return _(),P(O,E(z(r(N)(r(d),0))),{default:S(()=>[R,T,t("div",W,[t("span",{onClick:a[0]||(a[0]=(...l)=>r(o).nav.next&&r(o).nav.next(...l)),class:"px-2 py-1 rounded cursor-pointer",hover:"bg-white bg-opacity-10"},[B(" Press Space for next page "),c(v,{class:"inline"})])]),t("div",q,[t("button",{onClick:a[1]||(a[1]=l=>r(o).nav.openInEditor()),title:"Open in Editor",class:"text-xl icon-btn opacity-50 !border-none !hover:text-white"},[c(f)]),t("a",F,[c(h)])])]),_:1},16)}}},Y=m(G,[["__file","/@slidev/slides/1.md"]]);export{Y as default};
