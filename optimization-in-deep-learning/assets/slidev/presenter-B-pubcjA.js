import{_ as J,a as Y,b as Z,c as ee,d as te}from"../modules/unplugin-icons-D9Tg0CH2.js";import{t as V,d as R,c as b,i as f,B as t,o as i,z as v,b as k,e,l as r,F as q,x as N,g as z,_ as se,a2 as oe,a3 as ne,h as I,p as Q,a as U,E as G,N as le,$ as ae,K as re,a4 as ie,a5 as ce,k as $,a6 as ue}from"../modules/vue-Dim-dygt.js";import{j as de,b as P,k as pe,r as me,C as ve,u as _e,c as H,h as fe,l as xe,d as be,m as ke}from"../index-GjFdxGIJ.js";import{r as ge,u as he,S as K,I as O,Q as ye,a as Ce,N as we,G as Se}from"./SlidesShow-BMozc_QJ.js";import{s as $e,b as T,p as Ne,S as ze,g as Ie,c as Fe,i as De,d as Be}from"./bottom-DtJVD4uQ.js";import{N as Me}from"./NoteDisplay-CSBR0dd0.js";import Te from"./DrawingControls-KWZL2qru.js";import{u as Ve}from"./DrawingPreview-CsbPS-6g.js";import"../modules/shiki-D5uTpyAc.js";import"./context-B-08xyCf.js";function qe(a){var n;return{info:V((n=de(a))==null?void 0:n.meta.slide),update:async()=>{}}}const Re=R({__name:"NoteStatic",props:{no:{type:Number,required:!0},class:{type:String,required:!1},clicksContext:{type:null,required:!1}},setup(a){const n=a,{info:s}=qe(n.no);return(u,c)=>{var o,x;return i(),b(Me,{class:f(n.class),note:(o=t(s))==null?void 0:o.note,"note-html":(x=t(s))==null?void 0:x.noteHTML,"clicks-context":u.clicksContext},null,8,["class","note","note-html","clicks-context"])}}}),Pe=P(Re,[["__file","/home/runner/work/slide/slide/optimization-in-deep-learning/node_modules/@slidev/client/internals/NoteStatic.vue"]]),W=a=>(Q("data-v-52a29648"),a=a(),U(),a),Le=["title"],je={class:"flex gap-0.5 items-center min-w-16 font-mono mr1"},Ee=W(()=>e("div",{"flex-auto":""},null,-1)),Ge={"text-primary":""},He=W(()=>e("span",{op25:""},"/",-1)),Ke={op50:""},Oe=["min","max"],Qe=R({__name:"ClicksSlider",props:{clicksContext:{type:null,required:!0}},setup(a){const n=a,s=v(()=>n.clicksContext.total),u=v(()=>pe(0,n.clicksContext.clicksStart,s.value)),c=v(()=>s.value-u.value+1),o=v({get(){return n.clicksContext.current>s.value?-1:n.clicksContext.current},set(g){n.clicksContext.current=g}}),x=v(()=>me(u.value,s.value+1));function F(){(o.value<0||o.value>s.value)&&(o.value=0)}return(g,d)=>{const D=J;return i(),k("div",{class:f(["flex gap-1 items-center select-none",c.value?"":"op50"]),title:`Clicks in this slide: ${c.value}`},[e("div",je,[r(D,{"text-sm":"",op50:""}),Ee,o.value>=0&&o.value!==t(ve)?(i(),k(q,{key:0},[e("span",Ge,N(o.value),1),He],64)):z("v-if",!0),e("span",Ke,N(s.value),1)]),e("div",{relative:"","flex-auto":"",h5:"","font-mono":"",flex:"~",onDblclick:d[2]||(d[2]=l=>o.value=g.clicksContext.total)},[(i(!0),k(q,null,se(x.value,l=>(i(),k("div",{key:l,border:"y main","of-hidden":"",relative:"",class:f([l===0?"rounded-l border-l":"",l===s.value?"rounded-r border-r":""]),style:I({width:c.value>0?`${1/c.value*100}%`:"100%"})},[e("div",{absolute:"","inset-0":"",class:f(l<=o.value?"bg-primary op15":"")},null,2),e("div",{class:f([+l==+o.value?"text-primary font-bold op100 border-primary":"op30 border-main",l===0?"rounded-l":"",l===s.value?"rounded-r":"border-r-2"]),"w-full":"","h-full":"","text-xs":"",flex:"","items-center":"","justify-center":"","z-1":""},N(l),3)],6))),128)),oe(e("input",{"onUpdate:modelValue":d[0]||(d[0]=l=>o.value=l),class:"range",absolute:"","inset-0":"",type:"range",min:u.value,max:s.value,step:1,"z-10":"",op0:"",style:I({"--thumb-width":`${1/(c.value+1)*100}%`}),onMousedown:F,onFocus:d[1]||(d[1]=l=>{var C;return(C=l.currentTarget)==null?void 0:C.blur()})},null,44,Oe),[[ne,o.value]])],32)],10,Le)}}}),Ue=P(Qe,[["__scopeId","data-v-52a29648"],["__file","/home/runner/work/slide/slide/optimization-in-deep-learning/node_modules/@slidev/client/internals/ClicksSlider.vue"]]),L=a=>(Q("data-v-d0ba6bbb"),a=a(),U(),a),We={class:"bg-main h-full slidev-presenter"},Ae=L(()=>e("div",{class:"absolute left-0 top-0 bg-main border-b border-r border-main px2 py1 op50 text-sm"}," Current ",-1)),Xe={class:"relative grid-section next flex flex-col p-2 lg:p-4"},Je=L(()=>e("div",{class:"absolute left-0 top-0 bg-main border-b border-r border-main px2 py1 op50 text-sm"}," Next ",-1)),Ye={key:1,class:"grid-section note grid grid-rows-[1fr_min-content] overflow-hidden"},Ze={class:"border-t border-main py-1 px-2 text-sm"},et={class:"grid-section bottom flex"},tt=L(()=>e("div",{"flex-auto":""},null,-1)),st={class:"text-2xl pl-2 pr-6 my-auto tabular-nums"},ot={class:"progress-bar"},nt=R({__name:"presenter",setup(a){const n=V();ge(),he(n);const{clicksContext:s,currentSlideNo:u,currentSlideRoute:c,hasNext:o,nextRoute:x,slides:F,queryClicks:g,getPrimaryClicks:d,total:D}=_e(),{isDrawing:l}=Ve(),C=H.titleTemplate.replace("%s",H.title||"Slidev");fe({title:`Presenter - ${C}`}),V(!1);const{timer:A,resetTimer:j}=xe(),X=v(()=>F.value.map(S=>be(S))),p=v(()=>s.value.current<s.value.total?[c.value,s.value.current+1]:o.value?[x.value,0]:null),w=v(()=>p.value&&X.value[p.value[0].no-1]);return G([c,g],()=>{w.value&&(w.value.current=p.value[1])},{immediate:!0}),le(),ae(()=>{const S=n.value.querySelector("#slide-content"),_=re(ie()),B=ce();G(()=>{if(!B.value||l.value||!$e.value)return;const m=S.getBoundingClientRect(),h=(_.x-m.left)/m.width*100,y=(_.y-m.top)/m.height*100;if(!(h<0||h>100||y<0||y>100))return{x:h,y}},m=>{ke.cursor=m})}),(S,_)=>{var E;const B=Y,m=Z,h=ee,y=te;return i(),k(q,null,[e("div",We,[e("div",{class:f(["grid-container",`layout${t(Ne)}`])},[e("div",{ref_key:"main",ref:n,class:"relative grid-section main flex flex-col"},[r(K,{key:"main",class:"h-full w-full p-2 lg:p-4 flex-auto"},{default:$(()=>[r(Ce,{"render-context":"presenter"})]),_:1}),(i(),b(Ue,{key:(E=t(c))==null?void 0:E.no,"clicks-context":t(d)(t(c)),class:"w-full pb2 px4 flex-none"},null,8,["clicks-context"])),Ae],512),e("div",Xe,[p.value&&w.value?(i(),b(K,{key:"next",class:"h-full w-full"},{default:$(()=>[(i(),b(ze,{is:p.value[0].component,key:p.value[0].no,"clicks-context":w.value,class:f(t(Ie)(p.value[0])),route:p.value[0],"render-context":"previewNext"},null,8,["is","clicks-context","class","route"]))]),_:1})):z("v-if",!0),Je]),z(" Notes "),(i(),k("div",Ye,[(i(),b(Pe,{key:`static-${t(u)}`,no:t(u),class:"w-full max-w-full h-full overflow-auto p-2 lg:p-4",style:I({fontSize:`${t(Fe)}em`}),"clicks-context":t(s)},null,8,["no","style","clicks-context"])),e("div",Ze,[r(O,{title:"Increase font size",onClick:t(De)},{default:$(()=>[r(B)]),_:1},8,["onClick"]),r(O,{title:"Decrease font size",onClick:t(Be)},{default:$(()=>[r(m)]),_:1},8,["onClick"]),z("v-if",!0)])])),e("div",et,[r(we,{persist:!0}),tt,e("div",{class:"timer-btn my-auto relative w-22px h-22px cursor-pointer text-lg",opacity:"50 hover:100",onClick:_[2]||(_[2]=(...M)=>t(j)&&t(j)(...M))},[r(h,{class:"absolute"}),r(y,{class:"absolute opacity-0"})]),e("div",st,N(t(A)),1)]),(i(),b(Te,{key:2}))],2),e("div",ot,[e("div",{class:"progress h-3px bg-primary transition-all",style:I({width:`${(t(u)-1)/(t(D)-1)*100}%`})},null,4)])]),r(Se),r(ye,{modelValue:t(T),"onUpdate:modelValue":_[3]||(_[3]=M=>ue(T)?T.value=M:null)},null,8,["modelValue"])],64)}}}),_t=P(nt,[["__scopeId","data-v-d0ba6bbb"],["__file","/home/runner/work/slide/slide/optimization-in-deep-learning/node_modules/@slidev/client/pages/presenter.vue"]]);export{_t as default};
