"?M
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1fffff)?@Afffff)?@a2??k5???i2??k5????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff??@9fffff??@Afffff??@Ifffff??@aL{?????i# rb;???Unknown?
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1??????c@9??????c@A??????c@I??????c@a?,N?@??i?u ?l????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??????O@9??????O@A??????O@I??????O@a\~?b7??i????????Unknown
]HostCast"Adam/Cast_1(1??????M@9??????M@A??????M@I??????M@ag??`8??i?c???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1??????H@9??????H@A??????H@I??????H@a???lbU??i*Jɠ?Y???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333sD@933333sD@A33333sD@I33333sD@a֭?촻??i??|t?????Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1333333D@9333333D@A333333D@I333333D@a?3?V???i??M??????Unknown
o
HostSoftmax"sequential/dense_1/Softmax(1333333D@9333333D@A333333D@I333333D@a?3?V???i?*? ???Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1fffff?@@9fffff?@@Afffff?@@Ifffff?@@a?u ?l?{?i???aW???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1??????=@9??????=@A??????=@I??????=@ag??`8x?i???ч???Unknown
?HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1?????L<@9?????L<@A?????L<@I?????L<@a?<H(w?i??b?!????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      <@9      <@A      <@I      <@ao:?8?v?i?^W?????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1??????;@9??????;@A??????;@I??????;@ar???n?v?i<?4???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1??????9@9??????9@A3333334@I3333334@a?3?V?p?iE?'?-2???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff1@9ffffff1@Affffff1@Iffffff1@aKmZ?yl?i??2??N???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      1@9      1@A      1@I      1@aP??"?k?i?|U?yj???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate(13333334@93333334@Affffff+@Iffffff+@ar????kf?iC,8?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??????%@9??????%@A??????%@I??????%@a??u???a?i/?õ?????Unknown
iHostWriteSummary"WriteSummary(1333333$@9333333$@A333333$@I333333$@a?3?V?`?i4?w????Unknown?
gHostStridedSlice"strided_slice(1??????"@9??????"@A??????"@I??????"@a@???]p^?i1?Z;Q????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      "@9      "@A      "@I      "@aF3o??t]?i??N?????Unknown
dHostDataset"Iterator::Model(1?????YH@9?????YH@A?????? @I?????? @aUe?9z*[?i~?k??????Unknown
[HostAddV2"Adam/add(1333333@9333333@A333333@I333333@a^t??Y?i?ѥ?d????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@amM??=W?i??>????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @ao:?8?V?i?9q?w????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @ay?lGFU?i???????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1??????@9??????@A??????@I??????@a??W.?JT?i?r@???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?J@9     ?J@A333333@I333333@a???\??R?i??<S????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a???h??R?i???N???Unknown
^ HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a?o.t-TR?i#??e<#???Unknown
?!HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff@Affffff@Iffffff@a?o.t-TR?i[e|f,???Unknown
Z"HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a?%a?;?P?i?3<?4???Unknown
`#HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a???q]P?i`6??<???Unknown
a$HostCast"sequential/Cast(1ffffff@9ffffff@Affffff@Iffffff@aA?'ѓN?iN??tD???Unknown
?%HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@aA?'ѓN?i<??K???Unknown
?&HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1ffffff@9ffffff@Affffff@Iffffff@aA?'ѓN?i*?A?S???Unknown
?'HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@aV#?E??J?isw?8Z???Unknown
V(HostSum"Sum_2(1333333@9333333@A333333@I333333@a^t??I?iz~$К`???Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@ab?c???H?i`WG??f???Unknown
q*HostCast"sequential/dropout/dropout/Cast(1ffffff@9ffffff@Affffff@Iffffff@ab?c???H?iF0j?
m???Unknown
v+HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??????@9??????@A??????@I??????@ak???̐G?i꬘??r???Unknown
[,HostPow"
Adam/Pow_1(1      @9      @A      @I      @ao:?8?F?im??K?x???Unknown
e-Host
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@as????AF?i??9~???Unknown?
o.HostMul"sequential/dropout/dropout/Mul(1ffffff
@9ffffff
@Affffff
@Iffffff
@aw???E?iG9?????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_3(1??????	@9??????	@A??????	@I??????	@a|}?D?i0Ҍ?܈???Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1??????	@9??????	@A??????	@I??????	@a|}?D?iP??w????Unknown
V1HostCast"Cast(1??????	@9??????	@A??????	@I??????	@a|}?D?ipZV????Unknown
t2HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a?w?EU?C?iN?i?>????Unknown
b3HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?w?EU?C?i,*??'????Unknown
~4HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1333333@9333333@A333333@I333333@a???\??B?i?c??????Unknown
?5HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1333333@9333333@A333333@I333333@a???\??B?i??i??????Unknown
?6HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1ffffff@9ffffff@Affffff@Iffffff@a?o.t-TB?iB?ƭ:????Unknown
Y7HostPow"Adam/Pow(1??????@9??????@A??????@I??????@a??u???A?i??)ԥ????Unknown
`8HostDivNoNan"
div_no_nan(1??????@9??????@A??????@I??????@a??u???A?i8d??????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@a?g??A?i??;R????Unknown
?:HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a?g??A?i??]}?????Unknown
?;HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1??????@9??????@A??????@I??????@a?g??A?iFrƾ?????Unknown
X<HostEqual"Equal(1      @9      @A      @I      @a???q]@?i?4?????Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1333333@9333333@A333333@I333333@a9????k??i?F???????Unknown
w>HostReadVariableOp"div_no_nan_1/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@aA?'ѓ>?i?k#%?????Unknown
?HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1ffffff@9ffffff@Affffff@Iffffff@aA?'ѓ>?i????`????Unknown
?@HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@aA?'ѓ>?i|?J$????Unknown
?AHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1??????@9??????@A??????@I??????@aK???k?<?iR????????Unknown
?BHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1?????? @9?????? @A?????? @I?????? @aR?E.D~;?iu?-????Unknown
?CHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @aZ??\/:?i???s????Unknown
oDHostReadVariableOp"Adam/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??ab?c???8?i|:??????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??ab?c???8?i??ˠ?????Unknown
qFHostMul" sequential/dropout/dropout/Mul_1(1ffffff??9ffffff??Affffff??Iffffff??ab?c???8?i?T]??????Unknown
vGHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1333333??9333333??A333333??I333333??as????A6?i%e?ӏ????Unknown
XHHostCast"Cast_2(1333333??9333333??A333333??I333333??as????A6?iVu?X????Unknown
?IHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff?5@9fffff?5@A333333??I333333??as????A6?i??4= ????Unknown
}JHostMul",gradient_tape/sequential/dropout/dropout/Mul(1      ??9      ??A      ??I      ??a?w?EU?3?iv9ݧ?????Unknown
XKHostCast"Cast_3(1333333??9333333??A333333??I333333??a9????k/?ic?c?????Unknown
XLHostCast"Cast_4(1333333??9333333??A333333??I333333??a9????k/?i??Q?????Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??a9????k/?i??x????Unknown
yNHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333??9333333??A333333??I333333??a9????k/?i??Ŗo????Unknown
?OHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1????????9????????A????????I????????aK???k?,?iۅm<????Unknown
?PHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1????????9????????A????????I????????aK???k?,?i|?ED	????Unknown
TQHostMul"Mul(1????????9????????A????????I????????a|}?$?i?GlX????Unknown
aRHostIdentity"Identity(1????????9????????A????????I????????a|}??i     ???Unknown?*?L
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff??@9fffff??@Afffff??@Ifffff??@a??/M???i??/M????Unknown?
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1??????c@9??????c@A??????c@I??????c@a?yDYC??i2ҬW?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??????O@9??????O@A??????O@I??????O@a??bk??i???????Unknown
]HostCast"Adam/Cast_1(1??????M@9??????M@A??????M@I??????M@a???쯟?iC?hu?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1??????H@9??????H@A??????H@I??????H@aٶ?E.???i????_s???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333sD@933333sD@A33333sD@I33333sD@a?zv?Z???i?S???"???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1333333D@9333333D@A333333D@I333333D@a??
Rן??iϨw?????Unknown
oHostSoftmax"sequential/dense_1/Softmax(1333333D@9333333D@A333333D@I333333D@a??
Rן??i???1?|???Unknown
t	Host_FusedMatMul"sequential/dense_1/BiasAdd(1fffff?@@9fffff?@@Afffff?@@Ifffff?@@a H?ґ?i?v?????Unknown
}
HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1??????=@9??????=@A??????=@I??????=@a???쯏?iX???׉???Unknown
?HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1?????L<@9?????L<@A?????L<@I?????L<@ag???K??i??2???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      <@9      <@A      <@I      <@a??/oq???ix????z???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1??????;@9??????;@A??????;@I??????;@aw??ҋ??i??	B????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1??????9@9??????9@A3333334@I3333334@a??
Rן??ij?Q??G???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff1@9ffffff1@Affffff1@Iffffff1@a4?R-???i?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      1@9      1@A      1@I      1@a?L?2??ig?9?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate(13333334@93333334@Affffff+@Iffffff+@a??,U}?i?u#????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??????%@9??????%@A??????%@I??????%@a؛fd?w?iG?>*?C???Unknown
iHostWriteSummary"WriteSummary(1333333$@9333333$@A333333$@I333333$@a??
Rןu?i????o???Unknown?
gHostStridedSlice"strided_slice(1??????"@9??????"@A??????"@I??????"@a?[X?[?s?i>????????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      "@9      "@A      "@I      "@aށU~?Ds?iB3~km????Unknown
dHostDataset"Iterator::Model(1?????YH@9?????YH@A?????? @I?????? @a???kA?q?iN&V??????Unknown
[HostAddV2"Adam/add(1333333@9333333@A333333@I333333@a8J:4?p?i???V^???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a?>?Ogn?iÖg? ???Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a??/oq?m?iqƉؾ>???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??W?k?iI??/?Z???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1??????@9??????@A??????@I??????@a?$?iz?j?inb?? u???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?J@9     ?J@A333333@I333333@aF????h?iN{???????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1??????@9??????@A??????@I??????@a+Ol`hh?i???_????Unknown
^HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a??%??g?i[??Y????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff@Affffff@Iffffff@a??%??g?ig8?T????Unknown
Z HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a??`¦?e?i??1+????Unknown
`!HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aiW??ie?iY|?9????Unknown
a"HostCast"sequential/Cast(1ffffff@9ffffff@Affffff@Iffffff@a?_??c?il~;?F???Unknown
?#HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?_??c?i??R?(???Unknown
?$HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1ffffff@9ffffff@Affffff@Iffffff@a?_??c?i???ޫ<???Unknown
?%HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@ao=??q?a?i?%?P:N???Unknown
V&HostSum"Sum_2(1333333@9333333@A333333@I333333@a8J:4?`?i?o/??^???Unknown
t'HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a??Y?E`?it?3o???Unknown
q(HostCast"sequential/dropout/dropout/Cast(1ffffff@9ffffff@Affffff@Iffffff@a??Y?E`?i????x???Unknown
v)HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??????@9??????@A??????@I??????@a?ψ0??^?if?z?????Unknown
[*HostPow"
Adam/Pow_1(1      @9      @A      @I      @a??/oq?]?i=?2?ߝ???Unknown
e+Host
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@a[?֭3]?i?r	?n????Unknown?
o,HostMul"sequential/dropout/dropout/Mul(1ffffff
@9ffffff
@Affffff
@Iffffff
@a#i}??B\?i8??T?????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_3(1??????	@9??????	@A??????	@I??????	@a?F$+?g[?i[C1D????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1??????	@9??????	@A??????	@I??????	@a?F$+?g[?i~?*?????Unknown
V/HostCast"Cast(1??????	@9??????	@A??????	@I??????	@a?F$+?g[?i?g@??????Unknown
t0HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a~r?<?Y?i?????????Unknown
b1HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a~r?<?Y?i???%]????Unknown
~2HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1333333@9333333@A333333@I333333@aF????X?if\%?	???Unknown
?3HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1333333@9333333@A333333@I333333@aF????X?i???$3???Unknown
?4HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1ffffff@9ffffff@Affffff@Iffffff@a??%??W?ib?b?0"???Unknown
Y5HostPow"Adam/Pow(1??????@9??????@A??????@I??????@a؛fd?W?i?G?-???Unknown
`6HostDivNoNan"
div_no_nan(1??????@9??????@A??????@I??????@a؛fd?W?i?8?P9???Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@a?y?EDV?i???+rD???Unknown
?8HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a?y?EDV?ixFjN?O???Unknown
?9HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1??????@9??????@A??????@I??????@a?y?EDV?i5?;q?Z???Unknown
X:HostEqual"Equal(1      @9      @A      @I      @aiW??iU?ia?,?je???Unknown
u;HostReadVariableOp"div_no_nan/ReadVariableOp(1333333@9333333@A333333@I333333@a15[ ʍT?i??<ڱo???Unknown
w<HostReadVariableOp"div_no_nan_1/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?_??S?iVl ?y???Unknown
=HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1ffffff@9ffffff@Affffff@Iffffff@a?_??S?iכfd????Unknown
?>HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?_??S?iXˬ=????Unknown
??HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1??????@9??????@A??????@I??????@a????N?R?i?,T?????Unknown
?@HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1?????? @9?????? @A?????? @I?????? @a??O??Q?ivT?\?????Unknown
?AHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @aT??? Q?i???7????Unknown
oBHostReadVariableOp"Adam/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a??Y?EP?i??Z????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a??Y?EP?iVmo[}????Unknown
qDHostMul" sequential/dropout/dropout/Mul_1(1ffffff??9ffffff??Affffff??Iffffff??a??Y?EP?i<&?????Unknown
vEHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1333333??9333333??A333333??I333333??a[?֭3M?i????????Unknown
XFHostCast"Cast_2(1333333??9333333??A333333??I333333??a[?֭3M?ia'??/????Unknown
?GHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff?5@9fffff?5@A333333??I333333??a[?֭3M?i???v????Unknown
}HHostMul",gradient_tape/sequential/dropout/dropout/Mul(1      ??9      ??A      ??I      ??a~r?<?I?i???????Unknown
XIHostCast"Cast_3(1333333??9333333??A333333??I333333??a15[ ʍD?iRА?????Unknown
XJHostCast"Cast_4(1333333??9333333??A333333??I333333??a15[ ʍD?i?*????Unknown
wKHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??a15[ ʍD?i???sM????Unknown
yLHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333??9333333??A333333??I333333??a15[ ʍD?i?)?p????Unknown
?MHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1????????9????????A????????I????????a????N?B?i?~й&????Unknown
?NHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1????????9????????A????????I????????a????N?B?i1?w??????Unknown
TOHostMul"Mul(1????????9????????A????????I????????a?F$+?g;?i?M}?I????Unknown
aPHostIdentity"Identity(1????????9????????A????????I????????a?F$+?g+?i?????????Unknown?2Nvidia GPU (Turing)