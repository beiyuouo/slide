import{d as _,a0 as u,z as h,b as a,e as t,x as s,B as c,F as f,_ as g,o as n,a1 as v,l as x,g as b}from"../modules/vue-Dim-dygt.js";import{u as N,h as y,c as m,b as k}from"../index-GjFdxGIJ.js";import{N as w}from"./NoteDisplay-CSBR0dd0.js";import"../modules/shiki-D5uTpyAc.js";const B={id:"page-root"},L={class:"m-4"},T={class:"mb-10"},V={class:"text-4xl font-bold mt-2"},z={class:"opacity-50"},D={class:"text-lg"},H={class:"font-bold flex gap-2"},S={class:"opacity-50"},C=t("div",{class:"flex-auto"},null,-1),F={key:0,class:"border-main mb-8"},M=_({__name:"print",setup(j){const{slides:d,total:p}=N();u(`
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
`),y({title:`Notes - ${m.title}`});const l=h(()=>d.value.map(o=>{var i;return(i=o.meta)==null?void 0:i.slide}).filter(o=>o!==void 0&&o.noteHTML!==""));return(o,i)=>(n(),a("div",B,[t("div",L,[t("div",T,[t("h1",V,s(c(m).title),1),t("div",z,s(new Date().toLocaleString()),1)]),(n(!0),a(f,null,g(l.value,(e,r)=>(n(),a("div",{key:r,class:"flex flex-col gap-4 break-inside-avoid-page"},[t("div",null,[t("h2",D,[t("div",H,[t("div",S,s(e==null?void 0:e.no)+"/"+s(c(p)),1),v(" "+s(e==null?void 0:e.title)+" ",1),C])]),x(w,{"note-html":e.noteHTML,class:"max-w-full"},null,8,["note-html"])]),r<l.value.length-1?(n(),a("hr",F)):b("v-if",!0)]))),128))])]))}}),q=k(M,[["__file","/home/runner/work/slide/slide/optimization-in-deep-learning/node_modules/@slidev/client/pages/presenter/print.vue"]]);export{q as default};
