import{t as o,al as z,am as F,an as j,y as c,ao as A,ap as B,Q as t,S as _,R as E,aq as L,d as g,_ as x,o as m,b as C,F as P,g as O,e as T,Y as i,ar as S,c as $,h as D,i as N,af as R,A as M,as as q,v as I}from"../modules/vue-CbUuoOfz.js";import{f as V,_ as W,c as r,b as k,x as G,y as K,z as U,A as H,B as Q,D as X}from"../index-Da20ahjm.js";const ie=o(!1),le=o(!1),re=o(!1),Y=o(!1),ce=o(!0),de=z({xs:460,...L}),y=F(),ue=j(),ve=c(()=>y.height.value-y.width.value/V.value>120),fe=A(_?document.body:null),h=B(),pe=c(()=>{var s,e;return["INPUT","TEXTAREA"].includes(((s=h.value)==null?void 0:s.tagName)||"")||((e=h.value)==null?void 0:e.classList.contains("CodeMirror-code"))}),ge=c(()=>{var s;return["BUTTON","A"].includes(((s=h.value)==null?void 0:s.tagName)||"")});t("slidev-camera","default",{listenToStorageChanges:!1});t("slidev-mic","default",{listenToStorageChanges:!1});const me=t("slidev-scale",0),he=t("slidev-presenter-cursor",!0,{listenToStorageChanges:!1}),_e=t("slidev-show-editor",!1,{listenToStorageChanges:!1}),we=t("slidev-editor-vertical",!1,{listenToStorageChanges:!1});t("slidev-editor-width",_?window.innerWidth*.4:318,{listenToStorageChanges:!1});t("slidev-editor-height",_?window.innerHeight*.4:300,{listenToStorageChanges:!1});const p=t("slidev-presenter-font-size",1,{listenToStorageChanges:!1}),f=t("slidev-presenter-layout",1,{listenToStorageChanges:!1});function Ce(){f.value=f.value+1,f.value>2&&(f.value=1)}function Se(){p.value=Math.min(2,p.value+.1)}function ye(){p.value=Math.max(.5,p.value-.1)}const xe=E(Y);function Te(s,e=""){var d,n;const a=["slidev-page",e],l=(n=(d=s==null?void 0:s.meta)==null?void 0:d.slide)==null?void 0:n.no;return l!=null&&a.push(`slidev-page-${l}`),a.filter(Boolean).join(" ")}async function ke(){const{saveAs:s}=await W(()=>import("../modules/file-saver-EUMWMpoS.js").then(e=>e.F),[]);s(typeof r.download=="string"?r.download:r.exportFilename?`${r.exportFilename}.pdf`:"/slide/testslidev-exported.pdf",`${r.title}.pdf`)}const Z={class:"h-full w-full flex items-center justify-center gap-2 slidev-slide-loading"},J=T("div",{class:"i-svg-spinners-90-ring-with-bg text-xl"},null,-1),ee=T("div",null,"Loading slide...",-1),se=g({__name:"SlideLoading",setup(s){const e=o(!1);return x(()=>{setTimeout(()=>{e.value=!0},200)}),(a,l)=>(m(),C("div",Z,[e.value?(m(),C(P,{key:0},[J,ee],64)):O("v-if",!0)]))}}),te=k(se,[["__file","/home/runner/work/slide/slide/test/node_modules/@slidev/client/internals/SlideLoading.vue"]]),ae=g({__name:"SlideWrapper",props:{clicksContext:{type:Object,required:!0},renderContext:{type:String,default:"slide"},active:{type:Boolean,default:!1},is:{type:Function,required:!0},route:{type:Object,required:!0}},setup(s){const e=s,a=c(()=>{var n,u;return((u=(n=e.route.meta)==null?void 0:n.slide)==null?void 0:u.frontmatter.zoom)??1});i(G,e.route),i(K,o(e.route.no)),i(U,o(e.renderContext)),i(H,S(e,"active")),i(Q,S(e,"clicksContext")),i(X,a);const l=c(()=>a.value===1?void 0:{width:`${100/a.value}%`,height:`${100/a.value}%`,transformOrigin:"top left",transform:`scale(${a.value})`}),d=q({loader:async()=>{const n=await e.is();return g({setup(u,{attrs:b}){return x(()=>{var v,w;(w=(v=e.clicksContext)==null?void 0:v.onMounted)==null||w.call(v)}),()=>I(n.default,b)}})},delay:300,loadingComponent:te});return(n,u)=>(m(),$(R(M(d)),{style:D(l.value),"data-slidev-no":e.route.no,class:N({"disable-view-transition":!["slide","presenter"].includes(e.renderContext)})},null,8,["style","data-slidev-no","class"]))}}),be=k(ae,[["__scopeId","data-v-026ee359"],["__file","/home/runner/work/slide/slide/test/node_modules/@slidev/client/internals/SlideWrapper.vue"]]),ze={render(){return[]}},Fe={render(){return[]}};export{Fe as G,be as S,ze as a,Y as b,p as c,ye as d,ie as e,le as f,Te as g,ve as h,Se as i,_e as j,me as k,we as l,re as m,ue as n,ke as o,f as p,ce as q,ge as r,he as s,xe as t,pe as u,fe as v,y as w,de as x,Ce as y,h as z};
