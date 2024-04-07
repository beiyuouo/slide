import{d as _,$ as u,y as h,b as a,e as t,x as s,A as c,F as f,Z as v,o as n,a0 as g,l as x,g as b}from"../modules/vue-CbUuoOfz.js";import{u as y,h as N,c as m,b as k}from"../index-Da20ahjm.js";import{N as w}from"./NoteDisplay-D36rMc-F.js";import"../modules/shiki-B43ebrIS.js";const L={id:"page-root"},T={class:"m-4"},V={class:"mb-10"},B={class:"text-4xl font-bold mt-2"},D={class:"opacity-50"},H={class:"text-lg"},S={class:"font-bold flex gap-2"},A={class:"opacity-50"},C=t("div",{class:"flex-auto"},null,-1),F={key:0,class:"border-main mb-8"},M=_({__name:"print",setup($){const{slides:d,total:p}=y();u(`
@page {
  size: A4;
  margin-top: 1.5cm;
  margin-bottom: 1cm;
}
* {
  -webkit-print-color-adjust: exact;
}
html,
html body,
html #app,
html #page-root {
  height: auto;
  overflow: auto !important;
}
`),N({title:`Notes - ${m.title}`});const i=h(()=>d.value.map(o=>{var l;return(l=o.meta)==null?void 0:l.slide}).filter(o=>o!==void 0&&o.noteHTML!==""));return(o,l)=>(n(),a("div",L,[t("div",T,[t("div",V,[t("h1",B,s(c(m).title),1),t("div",D,s(new Date().toLocaleString()),1)]),(n(!0),a(f,null,v(i.value,(e,r)=>(n(),a("div",{key:r,class:"flex flex-col gap-4 break-inside-avoid-page"},[t("div",null,[t("h2",H,[t("div",S,[t("div",A,s(e==null?void 0:e.no)+"/"+s(c(p)),1),g(" "+s(e==null?void 0:e.title)+" ",1),C])]),x(w,{"note-html":e.noteHTML,class:"max-w-full"},null,8,["note-html"])]),r<i.value.length-1?(n(),a("hr",F)):b("v-if",!0)]))),128))])]))}}),Z=k(M,[["__file","/home/runner/work/slide/slide/test/node_modules/@slidev/client/pages/presenter/print.vue"]]);export{Z as default};
