<pre>
using device: cuda
loaded 261973 tokens
1 epoch = 63 batches
New Vocab Size is: 19248
TransformerModel(
  (token_embeddings): Embedding(19248, 576)
  (position_embeddings): Embedding(2048, 576)
  (layers): ModuleList(
    (0-29): 30 x TransformerLayer(
      (self_attn): CausalSelfAttention(
        (c_attn): Linear(in_features=576, out_features=1728, bias=True)
        (c_proj): Linear(in_features=576, out_features=576, bias=True)
      )
      (ln_1): LayerNorm((576,), eps=1e-05, elementwise_affine=True)
      (linear1): Linear(in_features=576, out_features=1536, bias=True)
      (linear2): Linear(in_features=1536, out_features=576, bias=True)
      (norm1): LayerNorm((576,), eps=1e-05, elementwise_affine=True)
    )
  )
  (lm_head): Linear(in_features=576, out_features=19248, bias=True)
)
Total trainable parameters: 105384624
</pre>
No checkpoint found at /kaggle/working/checkpoint.pth, starting fresh training.
<pre>
Training Batches 0-62: 100%|██████████| 63/63 [00:33<00:00,  1.87batch/s, loss=6.87]

Generated text at batch 62: He that will give good words to thee will flatter : , me ? with . what , , he , she to my he is ' And that good will my , ! know - the me : you as , , , ' , not I for his have not as my : , , the And ' I be , : : to have , the , is , to ? as the for , , is , what , , ! I . ' I , by . . is : you , ? ? the , And of have . : to and by ,
Training Batches 0-62: 100%|██████████| 63/63 [00:37<00:00,  1.67batch/s, loss=6.87]
Epoch Loss: 7.2082, Time: 37.68s
Training Batches 63-125: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=6.81]

Generated text at batch 125: He that will give good words to thee will flatter me a your , : I , have have so , I a , a by have not : And ; my is , : him you ! is : ; I ' be him to with he I , . , ? a , , but ' , ' and , : she - I she , this . I . , me you my not , , of and , are To : him with ; what is , she . is in good , be sir my to : , , to that ' , . :
Training Batches 63-125: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=6.81]
Epoch Loss: 6.5409, Time: 39.62s
Training Batches 126-188: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=6.8] 

Generated text at batch 188: He that will give good words to thee will flatter you not of . with you have no . And not : , it not s no the , have the you ; ; your d be will , ' his him the , as , it of she : , but to so , ? not I his , I ; a I of him ? , ? , you . . the , ' , it ; , : she have as of ? I I , not , ? , his , , me : , ' : : , I good ! : , he not
Training Batches 126-188: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=6.8]
Epoch Loss: 6.5201, Time: 39.62s
Training Batches 189-251: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=6.78]

Generated text at batch 251: He that will give good words to thee will flatter a this I , ; , do I : : . his and it you , . as , is you I for is : ; a a : , : it , : you , me him you , for , : a no , , a , And her . , his him s ' your in I , . for , . the is her to sir of your my to of . the , , , my ' , : not a for : to ' of ? he be a ? , , and ,
Training Batches 189-251: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=6.78]
Epoch Loss: 6.5118, Time: 39.72s
Training Batches 252-314: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=6.76]

Generated text at batch 314: He that will give good words to thee will flatter : it , is : . I the but . , ; by And so to I your him I and . to it I , me my my , . ? ' , her . your of , I will so , ; the : I , with so you , a to not to : : , , , , a . , you you . have - . do what , . s of , not ' this , ; is you , ' ' so , my : . : . : a the it my
Training Batches 252-314: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=6.76]
Epoch Loss: 6.5066, Time: 39.74s
Training Batches 315-377: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=6.73]

Generated text at batch 377: He that will give good words to thee will flatter it his of as , with is the to , : I . a with , , , you : be ? the him me : : , to And , is the , , s be is me thou ' my ? , so be ; ! : no , I that him I , you d I me , his the s thy d your : ? , is to , for you that it be to , that a , ; ! ll to me , for ? s . : but , I not . :
Training Batches 315-377: 100%|██████████| 63/63 [00:39<00:00,  1.58batch/s, loss=6.73]
Epoch Loss: 6.4874, Time: 39.77s
Training Batches 378-440: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.73]

Generated text at batch 440: He that will give good words to thee will flatter To : , ; , s , : your he I not your my me , the . : ; ! it and your I . to , , ' ' : , I , , not , ; in , it he this , ; do by : , : . ' you my : thou ! And , is . a , shall for to and , this it , : : ' to you for and , , I her the , , , , to : , of . . of . , ' your by
Training Batches 378-440: 100%|██████████| 63/63 [00:39<00:00,  1.58batch/s, loss=6.73]
Epoch Loss: 6.4802, Time: 39.81s
Training Batches 441-503: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=6.72]

Generated text at batch 503: He that will give good words to thee will flatter d of ll that you and To ! so , ; . of for it : and , thy , a my , to , is him to s d . him with : me not do you of you . and as this for ! he : you , you , I : to to for ; : so thou : for , , you - of . it thou to her , you your s - , . , have : : to : , . , , in ! the to is the you you a :
Training Batches 441-503: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=6.72]
Epoch Loss: 6.4786, Time: 39.72s
Training Batches 504-566: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.72]

Generated text at batch 566: He that will give good words to thee will flatter : , , , by , : and of And your shall . I s , this if ! ll I to : that d ' : : I be to with To I . a d me ; . . ' And me : his I your do , , as : : , for ? I ' , . in a it this , thy : , you be : , , him ' , s : , is , ! it have a the , : for . with . the your and ' I - it
Training Batches 504-566: 100%|██████████| 63/63 [00:39<00:00,  1.58batch/s, loss=6.72]
Epoch Loss: 6.4775, Time: 39.79s
Training Batches 567-629: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.72]

Generated text at batch 629: He that will give good words to thee will flatter ' as , be no it . , ! , I . , . : with be . that the is ! , this ! - you this not by , , with , and thou not : ! , I . that and him , , not ' I ll : not you : , . not , a of it of to : thou if , : thy : shall the it you , me . ' is , if you : . And his d it And for . I me , me . To I .
Training Batches 567-629: 100%|██████████| 63/63 [00:39<00:00,  1.58batch/s, loss=6.72]
Epoch Loss: 6.4766, Time: 39.77s
Training Batches 630-692: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 692: He that will give good words to thee will flatter the ; , the have of with him and me And with s all it be in : ' : I : ' so I with thou my for . : : the ; ? have him you . my be me it : with , d And his : his : : as with : thee : you of , , not all you s s I in his my : but his And , ' and of I ' : : be ; I ! . , not , ' ' be you this and To be the
Training Batches 630-692: 100%|██████████| 63/63 [00:39<00:00,  1.58batch/s, loss=6.69]
Epoch Loss: 6.4660, Time: 39.83s
Training Batches 693-755: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 755: He that will give good words to thee will flatter , that a s : with to ! is her to , s ; d : him you it you for . , : the that of ! : : by of . be have my , of of of . ? and . : , . . a ' And : the that my me : as ' : . . , the by : as that : me ; but . to . The ! to him . ! but ; the no you it you . I d not : . have you as you d ,
Training Batches 693-755: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.69]
Epoch Loss: 6.4638, Time: 39.35s
Training Batches 756-818: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 818: He that will give good words to thee will flatter . : , have . , , shall , a not , thou : , ? me , thee thou not ! s a to we : , will , and , do d to for to And is to . I ; of And d , , not , : a , , I we . and , , : my do me . And you , have as . ; so . this , shall my for have of , the you , I . , . not , the not and , : not have : :
Training Batches 756-818: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.69]
Epoch Loss: 6.4635, Time: 39.21s
Training Batches 819-881: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 881: He that will give good words to thee will flatter , , And this ' And for , not do : by ; her for ' ? have , ' s a and we for as s : a s , the with the : I in will him ; s ? , ' ; . it - ! my ' his and , the : a , - and s ; : , , ? , And . and that thy not . be my be The . the not to to not all , be , , , To ' . all to in to for , you
Training Batches 819-881: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.69]
Epoch Loss: 6.4633, Time: 39.22s
Training Batches 882-944: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 944: He that will give good words to thee will flatter s thy , : and have ' have of The , ' ! the you . as that be , you of ? do that of , of with , : the - ' ? you to . and ' : be . To you in the him and the with , have . and : in be , . you as that : d to . a , do . that thou to s ' a not thee we your : , ' , ; of ? , ? : the thy : and , the , my be
Training Batches 882-944: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.69]
Epoch Loss: 6.4631, Time: 39.26s
Training Batches 945-1007: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 1007: He that will give good words to thee will flatter that this , but ; . the ' : ; I the : the , d ! in do the : s . , . it you , the so of : , , ' my I the I of his , is of the ; thee - you ' . : : of , he the I The , but your so and as : ! thou : you ' I so , : but you I ! no of ' is and of . ; . ' is the I : to . his be we . And
Training Batches 945-1007: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.69]
Epoch Loss: 6.4544, Time: 39.26s
Training Batches 1008-1070: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 1070: He that will give good words to thee will flatter do to so as to . ; your ' to . my ? and to to , , , to , me be thy ; her I me , as . I ! not ' , ' . , to ' me : , him his have : , thy ' . I your in of , s and , , ' : . by not he I the , I The ' I . ? . ' . I that to is as of a and but , for it it , all s : with not d .
Training Batches 1008-1070: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.69]
Epoch Loss: 6.4537, Time: 39.30s
Training Batches 1071-1133: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 1133: He that will give good words to thee will flatter a , , the him this . to you my shall , my the , her I for his a , to , to , of a with my ? - and I have he this in ' that the I , . him that is a me do , his I , And is s that a , that ' ! is it , : , ! your you so with and : the . shall to your ? me : : ' : all And to ' that , , by , the . of : To him
Training Batches 1071-1133: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.69]
Epoch Loss: 6.4535, Time: 39.23s
Training Batches 1134-1196: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 1196: He that will give good words to thee will flatter ? of ' for ' you ; that not - : this as , , we , thy : be , s the , : his and in : so ; ' s with but no his d , ' thou thou . me it my by in to so . ' , in in , he , me you do , ! , is , ! the to : d : thee I , ? be , with be for : this , ; , . shall is , : as ! not , : of , : ;
Training Batches 1134-1196: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.69]
Epoch Loss: 6.4534, Time: 39.22s
Training Batches 1197-1259: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.69]

Generated text at batch 1259: He that will give good words to thee will flatter is : ' ' thou of , . , this he ' with : ' I , : . - : I no with his , : , will , me the . the have I not a him of the ; - to . : . a . be , of to ' I , , , , this , : as , have and To for . shall . ' your d to d that you but ' And be ; a him I : my : as I not that that , your thy to , ,
Training Batches 1197-1259: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.69]
Epoch Loss: 6.4533, Time: 39.28s
Training Batches 1260-1322: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.68]

Generated text at batch 1322: He that will give good words to thee will flatter and . be And , - ! my you , his as ' thee you and as he To to ! thy you my , his a : thou me s to s ? : I I his And thee : ; , d The I of ; , but ; : , me your he ' with ' to my are : the a ' And for have him ' : . for him , to to to to : and , the to the ; : you : ' to , , , the her , , And
Training Batches 1260-1322: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.68]
Epoch Loss: 6.4469, Time: 39.25s
Training Batches 1323-1385: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.68]

Generated text at batch 1385: He that will give good words to thee will flatter of ! for : ' ' I ; thee I you , : , with , : me it the and . the : : we , me . ' in ? I we the , of that : : the ' : and will not the to ' this a ' not , , a the the : . so this her is I be , we thee , : : that it : s ! , . : all d all I you , you me , of And ' , d and it all that I you
Training Batches 1323-1385: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.68]
Epoch Loss: 6.4465, Time: 39.18s
Training Batches 1386-1448: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.68]

Generated text at batch 1448: He that will give good words to thee will flatter I your , ? he of , , of the : ' this to the my a that ' and ? ; the will . ' your ' I s but as are I d with and for is to in my to , be ' . to ; , I , is To , the , : to for by , , is as ? ; me ' , ; . : the you , . . ' ' I and , ? ! you that to I ' : that I , - ; of , of .
Training Batches 1386-1448: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.68]
Epoch Loss: 6.4465, Time: 39.17s
Training Batches 1449-1511: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.68]

Generated text at batch 1511: He that will give good words to thee will flatter with ' his him ! : . it of and I ' the And , you : , in to ' , to not to I ' a you of and , is , in : ; this , he - this : to . . but the : will : , ; ? , , shall the , , . me thy : : the , be ' . so this thou And - - And I , ' not this I : , ? not d but ' : : is . , - do I by ,
Training Batches 1449-1511: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.68]
Epoch Loss: 6.4465, Time: 39.27s
Training Batches 1512-1574: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.68]

Generated text at batch 1574: He that will give good words to thee will flatter the is this To a . . : ? and to : . all : of you have that the is , , the to the be in ! . to , my to to I I thou for ? I , : ' . ? ' be I are you I , , ' I is ' be . - and ? , , and . - d ' I and ! this this do your you . have : : s is be to to : with , to he of her , ? I : as his
Training Batches 1512-1574: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.68]
Epoch Loss: 6.4465, Time: 39.33s
Training Batches 1575-1637: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 1637: He that will give good words to thee will flatter so the is , : ; , ! , you . the his to shall it I , ; in . To . in by ' and , ! me I And . not , , that . to with I ' thee as ' and and , ; d that do me thee ' by it of : : the not that : , , , , ; is have : a . . ' her not is : And : - that , ; and : of ; ! a : To , ? have . , '
Training Batches 1575-1637: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4425, Time: 39.22s
Training Batches 1638-1700: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 1700: He that will give good words to thee will flatter , in : is : And and to ; ' . my and . ! the To I a , d ' , a . d he for with ' to of by , this , I it : , is : that , for my I ; the ; are the as , , . thy , her ' of me , a this with it thee my as , be your with to . ' ' . . to I I the of to I And in . me you : , - your , . be .
Training Batches 1638-1700: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4424, Time: 39.26s
Training Batches 1701-1763: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 1763: He that will give good words to thee will flatter - , : the so my my the . the in and , I I a of ; my thy , I be with all shall ' have The of to . , by the , shall , , ' s ' and the s , in to as her my : so as ' will we thou ; ' , And be and . . : with this for not : . for : , , . ; : ' , of to ; I ! . have not not you your with you s it , : .
Training Batches 1701-1763: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4423, Time: 39.22s
Training Batches 1764-1826: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 1826: He that will give good words to thee will flatter I you I but , you ' . in but to , thou : the , : but shall that ' you of you the of in and - , . ' by , my and : this I of ' my this for , is it you my The for : ! , , as will ! in I for a in : . do and , is , , is thee not - of that . the do in , , , and . you a , , I d the - I , , , : the
Training Batches 1764-1826: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4423, Time: 39.32s
Training Batches 1827-1889: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 1889: He that will give good words to thee will flatter , his . , my , : his your I . have ! my him . , of I my ; your s I is I d - , to : : ! To will me thy . the : , is my . , your of to you with ; shall I : my : : I not as her have my to not thou , : , not , : : . her my ' : . to of my . for so ? s of the , , - for that it ; that a . '
Training Batches 1827-1889: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4423, Time: 39.26s
Training Batches 1890-1952: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 1952: He that will give good words to thee will flatter : , ' . and thou . ' , ' ' so : s , my he not and ' my , . : thy in And , , ? , of , , you : the To , ! a the ; but , that the his . this s that , is ' ' to , of , with , me and I thee that a his him And to the , thou - . : , have it , , : with he I : with of is thee d to to not him so : d
Training Batches 1890-1952: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4398, Time: 39.24s
Training Batches 1953-2015: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2015: He that will give good words to thee will flatter ; by The ' a he , , a s . your . the with ' : in be to the thou the : . the but , ? ' the To s . I a ; , not : , d but as a it I : him : ! , the : and me her The : by , : The ! will him : by ! and is ' and s : , ; . do your s : , ? and I a : he ' and ? ' ' , to And he . ,
Training Batches 1953-2015: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4397, Time: 39.24s
Training Batches 2016-2078: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2078: He that will give good words to thee will flatter , s , : in : , and me ' ? To the , , And and him as , is your are do of will of . in you s d I his all , , I you to : . have the my you have . ? , ? his by have to her . : her with , I I his s , And the your , his is and ; ' will , it to ' ; . that : . not . this for , him of , the the in with to , he
Training Batches 2016-2078: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4397, Time: 39.22s
Training Batches 2079-2141: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2141: He that will give good words to thee will flatter not , : and And ' with that : : but , . ' , , with I , the and in you the , , is my , we : : , : , with And I to of thou do to : of and your is be : the to . . And me this him his ! of thy . I , And are the thee his , s of to her will him , s s To ' is ; I and : the is but , ? . ' , . To I me thy
Training Batches 2079-2141: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4396, Time: 39.23s
Training Batches 2142-2204: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2204: He that will give good words to thee will flatter and ; be , he s you : ; and shall , my . , so , ' s that . ' s ' for I that you ? have this but but , ' we . not : . , by in of to the with : will of it : thy and my the and this as : he . d . ; all his that ' thou to : but ; this And all me he : ' . as ' , ? be s , , not thou ' to do not . The ? shall
Training Batches 2142-2204: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4396, Time: 39.20s
Training Batches 2205-2267: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2267: He that will give good words to thee will flatter my her do ; s in , ' , to : to ? to your but ' it your and ' thy , . the ; but shall . ! ' by ' , The the . , the : , , : the you not her ' of , my not . ; me be all and thy ; in , . - ; , a be ' a my , , , ' s - his . , , of and have ' thou : of my . not a with of I with , s and is
Training Batches 2205-2267: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4379, Time: 39.27s
Training Batches 2268-2330: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2330: He that will give good words to thee will flatter your with : you to I the ' her all : a it And , . The ' . , , I : , my , as to : . not And , ' : , and his d it To of ' he thy this s with . s , : : . him d so . this ' and are will of ; : the : to , : my And ! a ' to the and he the , a the ' , the and in , . will be : my it I and a ,
Training Batches 2268-2330: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4378, Time: 39.21s
Training Batches 2331-2393: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2393: He that will give good words to thee will flatter thou for not my that him ! not it ; the he ! he of : , my , shall ; it ' in . , of , , ' to . of : of ? : thou your of ! but . I ; : . his he ? : : . . you ? : : . ' : . : are ' in ' the this , And , . And ' , , of and to s I a and : - , s , ? for ' . I my , and I I my
Training Batches 2331-2393: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4378, Time: 39.25s
Training Batches 2394-2456: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2456: He that will give good words to thee will flatter : his . I I ? to : , , this . a , I ; , , to in ' I ? . that the and me the , of The ; . for , the to : a : my that have to ; d we a : ? . : of the ; shall in the as , , , be . that ' : ' I I , , to in . , to the , this you . . , of you me with be . with , all be in : the s you
Training Batches 2394-2456: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4378, Time: 39.22s
Training Batches 2457-2519: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2519: He that will give good words to thee will flatter ; , : I for s , thee I a with To , , thou the to this , and ! this his I to and are to is the d , and thou in my . have me for that and the d shall me shall . , so . ? . : and ' . a . : he of the . : to you , a : by have , And as I : To d his as the : To : ; we your and ' him . , thou your in that . thy ,
Training Batches 2457-2519: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4378, Time: 39.21s
Training Batches 2520-2582: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2582: He that will give good words to thee will flatter you , : he the , ' is . to for have and the be to to , that , so , a ' I : ' : ! I the but all not your to that , the your of shall his him , : ' all by your to . the , me : ' , , the ! , ' . d will , The ? ? And , , . you I to I . will : ! the I have : ' that in to ! his . , and ' and s , the
Training Batches 2520-2582: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4366, Time: 39.19s
Training Batches 2583-2645: 100%|██████████| 63/63 [00:34<00:00,  1.86batch/s, loss=6.67]

Generated text at batch 2645: He that will give good words to thee will flatter be to your as and s is as , ' and thee : my are thy The : : your have : my ! me , in I ' And ' , me ! , is ; : , ; ! the . thy I we , him of . : is I the that the ' , ; I ? in To is your ' it , ; you s will : the , , that , ! : that and of the will have , , , is his ; , to I . thee and to have
Training Batches 2583-2645: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4365, Time: 39.30s
Training Batches 2646-2708: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2708: He that will give good words to thee will flatter be ' you the but you me have your , you To thee ; me : s , I , you ; me is a is by and you , , , I of for my are : with ? . the be , ! my I , . you to ' me , the d a , . that , - you The : ' : as I ' : . : The have I And To shall are . : . of ' , ? . the your And , thou , , and And And my a
Training Batches 2646-2708: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4365, Time: 39.19s
Training Batches 2709-2771: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2771: He that will give good words to thee will flatter in a to his in , not I the thou it , the this ? I To a , ' ' ' . , : , ' in s of will , me , s of and , ! . that I him to a it , that ; the the this I . The that but your . . ' , of I . a the her this and ' , as to for , in ! the is a is , , , ' to , ; ; ' : s I And , To so you ,
Training Batches 2709-2771: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4365, Time: 39.15s
Training Batches 2772-2834: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2834: He that will give good words to thee will flatter . with . as thou ' ; me I , , be . him , s do : that ' for . have , : : he : . do , , be , ; , ; , , of , it to I the we I , And , ' the him him s : with : , not : ; , my thee him ' , ? : for : ; - : , : , I thy ; your my your , ' and . , this this I and so the will the of to :
Training Batches 2772-2834: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4365, Time: 39.19s
Training Batches 2835-2897: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2897: He that will give good words to thee will flatter . . . , : . : my And your , and . the , you And are s , , ! . my to ' do s so ' . , the that me , ! a not ' , : . my that a I ! . : , by for by ' - . shall and my the to he , s is that , s , in my not and , a but the - : is ' ; The ? I , , you . you it , ' with shall ' , ' :
Training Batches 2835-2897: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4356, Time: 39.31s
Training Batches 2898-2960: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 2960: He that will give good words to thee will flatter : we ? , we your ' the ? : I the . : you the is , him : ? d me , , . , and as her we . ' , ; shall , . , . have him , , in , he me I to - the s I ; ; a . will , not to , it a s thou this ' , in , the . ? - . , my , I the , I my my To the for your ' in to , ? his , , it of
Training Batches 2898-2960: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4356, Time: 39.15s
Training Batches 2961-3023: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3023: He that will give good words to thee will flatter I you ? : ! ? that not , : ; a not thy her do to : with of , me of my that , he are of s ! a and , s in this , . : , the , , , ? ; thee : , . : . but thee I my and so . my : : ! : , of all the to . is your ' the , I you ' of of and my with and , be my me ' to ' ! , I , : for and ,
Training Batches 2961-3023: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4356, Time: 39.17s
Training Batches 3024-3086: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3086: He that will give good words to thee will flatter . your and - d and ' ! with to thou To you : . I , d as he to and ' , to To : , d , that my ; and : my be : his . , ; ; . to , to And ! with me . ' , ' ' in ' a ! , have ' my do , a a as The . , to this , it , I , my : his ; you as do I to ; and me ? you for , be to to be :
Training Batches 3024-3086: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4356, Time: 39.17s
Training Batches 3087-3149: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3149: He that will give good words to thee will flatter ' , ' be ; ! ; to his And . all , - , . you ? the ' , ' . ' I : ? have the , , he . but my to ' a her the ' for and I , her The I ' I , , not : , the of , is ' : , : so ! : thee to ? in : , ' , is . : : a be : all , , to and , ? ' , a : , . do ' thy , is he
Training Batches 3087-3149: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4356, Time: 39.27s
Training Batches 3150-3212: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3212: He that will give good words to thee will flatter . , I in , ; not . have but . , , a , ! I be d but I you your my ! : . to the To not all me And , ' I your thee . of s a of the ! , he do , , in your he a , ? but , not that ' And to to d ; : s I all your in have be it for my , : my me The : : . is ' me be is , the that d I s ! I .
Training Batches 3150-3212: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4350, Time: 39.16s
Training Batches 3213-3275: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3275: He that will give good words to thee will flatter in me so a all ! ; thy have and . : I d , , have ' ! the , : your is that : as of , , his the . , : : I , it , , : : , are me my by that all so we is , . as him to thee and . : that thou with to . my ? my it be ! ; of the - he my you . the in , shall the as , And thy we the with the : his a ? have your
Training Batches 3213-3275: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4350, Time: 39.22s
Training Batches 3276-3338: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3338: He that will give good words to thee will flatter ' my ' s ? ' , And ? he so this in with . The as , ; I and are ' . : To ; , , you the ! And : so the the : , ' The . , and your , with ! To a your of , . ? I of to thee of be s I that , of me ' of you of . to is my the I ? - , be and are I - ' do I be : with is ! ! : is be , and d
Training Batches 3276-3338: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4350, Time: 39.16s
Training Batches 3339-3401: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3401: He that will give good words to thee will flatter , with you : . is for . to a And not of by be ' he to you , his : ; thy I for and her ' ! of me this his you me s thou it his , , and : be a ; ' a you a all d . he and have , of that ' shall your for , , and for ? so , ; : . s I s : a you you that ? to and to I and thee , To . ; to her . of . in the
Training Batches 3339-3401: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4350, Time: 39.31s
Training Batches 3402-3464: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3464: He that will give good words to thee will flatter : will ; a are me I ! : ! me thy the you , my have : is I and : as to the of is . d a I he ? ' I in . be : as not that to ' and thou so to : him by is we in not for he ; And it I : : : : ' To to to with s ' , ! as is it - , : thee I . ; : to ' he , a a I , be the , . your ' the
Training Batches 3402-3464: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4350, Time: 39.25s
Training Batches 3465-3527: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3527: He that will give good words to thee will flatter ' ? , , you the have . ' and and have : ; . her a ' of we your ' a s ? , not that , . : . to , me ' , I thee , my ? , I . thee to and , To so be : a ; in s : d , , to s ! the you The is the . and And my me that thou be the in d I ; , I s , not all And the and . I shall , I . . thy this
Training Batches 3465-3527: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4346, Time: 39.23s
Training Batches 3528-3590: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3590: He that will give good words to thee will flatter s : ! : - have , , her this will by . me : The : is is the and is : you that , ; I for , , I And his and . , d to To you , and are I me the a to her not I for my : and have , this thou : but ? your a a a my you , but , you with it he will : the . you : , with the and of and ! his it . ; , the ? : that the !
Training Batches 3528-3590: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4346, Time: 39.26s
Training Batches 3591-3653: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3653: He that will give good words to thee will flatter , I , , I that ' the : : thy thee , are : his with that me s of you to , , , in : with will all ; , And : . to , ? you me s , the . will is me to the to , in the : . by will to my , I , to , , and ' it ; , in you him And , , that ! but ' thy I : be ! this : d ? I s it d . to , your , .
Training Batches 3591-3653: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4346, Time: 39.27s
Training Batches 3654-3716: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3716: He that will give good words to thee will flatter I shall thee his your as have the : is you , : . : , the the my is this : as , , : : you , : ' - my thou And of thou , , , And . : all the my ; ' the in To with will you . him : , : s the , , the to shall you be - me : : , of my the , my ! the his , it we be for , ; , I To and I , thou The ; , my I
Training Batches 3654-3716: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4346, Time: 39.24s
Training Batches 3717-3779: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3779: He that will give good words to thee will flatter have : , ! , . with of : , , I with with : your and thou , to be , , I this my , and , ; . thou s , I the , we and : , I the , the him all the ; all And , of by my her And d you not ? will ; . ? , is ? my s , thou . : to ! be this the so in so of your will , he ' ' ? : with this , not this : . , :
Training Batches 3717-3779: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4346, Time: 39.17s
Training Batches 3780-3842: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3842: He that will give good words to thee will flatter : be and : for ' And I . and I his , to be to ; I : of ; thou your a I ! , the s the . the not not will ' ? . the to we . and in . the , and : : : ; thee ' for thy . ; be are your and shall not not is that it by : , - : the me , ! , ; ' my , thou he , : but . thy to him my , I ? - : , ? of
Training Batches 3780-3842: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4343, Time: 39.24s
Training Batches 3843-3905: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3905: He that will give good words to thee will flatter the that he ' I ' this , . , I thy her for , him And , , to , my the thou , is , . you with the , , that . with ; your you ? . my ' . that . ? his his ' this ! will ' the , to that but ' ; , the the in , my and , : , all , , a of be the . he : . by , of , : ' for s thy it you . : the your the a your
Training Batches 3843-3905: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4343, Time: 39.19s
Training Batches 3906-3968: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 3968: He that will give good words to thee will flatter : ' he ; the And ; : : this . and we as , : . To a d , of , the : his but : ' but this are and the , , . . of . my , but as him not with ? have not . ' but , of s . not ; to you that your by ; a thy , : d , I is with ! the , with . in d so my I , you : , with , . for with he the that ; ; , a
Training Batches 3906-3968: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4343, Time: 39.28s
Training Batches 3969-4031: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4031: He that will give good words to thee will flatter this , And him but ! your , : : And The thou thy , : but : your have with shall , thou ? as . the him , The . to to : not my I : be , , and to have be . . the . , the of ' of - he the : The he . , but ? . . d to his to , s have my I his for of ; : a a s that of ; the of d the : as have with be . I ? but
Training Batches 3969-4031: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4343, Time: 39.19s
Training Batches 4032-4094: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4094: He that will give good words to thee will flatter do ; in , , thy , ' by , in your , ; , is ; : , , not the , . have thee will s : that : , , , ' but ' , , : will , it I , ' : , that for , The . in will to And of , be ' for with in and , this me me , , And s : ? And to : do my for ; I be , : him : ! , his , : with him , , the all '
Training Batches 4032-4094: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4343, Time: 39.22s
Training Batches 4095-4157: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4157: He that will give good words to thee will flatter his not I , , ; to thee by . - ! all it , thy I ; of all and have I of ' , To , , you , be all , in this And ; a , the thy thou him , , . , he , To all my have , all but . ' , of thee , ' for of with , you . her to : To of him and . in not ; , : of , the , And I : : by not , . , : he the his
Training Batches 4095-4157: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4342, Time: 39.24s
Training Batches 4158-4220: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4220: He that will give good words to thee will flatter : ' ' I my , are , are the . , I The : , thou her , thee And ? , . . your the I , , and in of his that thy but , of the it do , I : . my , , to that , he : , we : of And you this I we , ' , . thou with : are ' the , of : to And . a you a d so to to , he . so : your . shall a a I , his !
Training Batches 4158-4220: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4342, Time: 39.31s
Training Batches 4221-4283: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4283: He that will give good words to thee will flatter shall are will he be ? , is have , and , And shall have in the , , to your this of : ? thee : be , not me and ; that but but in : will . to me ? , to , shall ; in that ; to : your to do , by The in and the ' my s be . I , to you and his for and , , : : as , , , you in thou the this s not , ; . will my not you ; me .
Training Batches 4221-4283: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4342, Time: 39.20s
Training Batches 4284-4346: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4346: He that will give good words to thee will flatter ; I of . ' ' the with are the him this d , , , s . the thy I his all it my : have : it in . : , by ? I will my d , the will it of in . , : you ; will her ! , with : I and the ; in thou : I , is is a to shall . ' And ; her in is is . d ! to . I . not ' to : will I he do : the I , shall be '
Training Batches 4284-4346: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4342, Time: 39.25s
Training Batches 4347-4409: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4409: He that will give good words to thee will flatter of that , : ' my with I with have , not he you and so my . , it , you : ' he by ? to , : of that ? I : . a thy you and not . ' , ' his as , , that and : , ! the I : me d . , - to the . the ' to thee , , have , . my a , so , will ! . : : I , d have this in it ' the d ? d . : : as
Training Batches 4347-4409: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4342, Time: 39.23s
Training Batches 4410-4472: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4472: He that will give good words to thee will flatter ? ! is ? : : I . a your , the with And thou not s a me : , , we the . the , , and her that , your I but ' ? , , , and : is his , of , the ' of d : and to him he to that the ? d the , , d : ' ' : , but . thy I do with that of have s : To , , a of . : , ; we be will s : all - a my I
Training Batches 4410-4472: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.29s
Training Batches 4473-4535: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4535: He that will give good words to thee will flatter , but ' . , but ! it to , d , have the with And ' his , . : , of to , the . , The ? that the s I but The . ; my , thy of , ' of , , but me the , it you have , he so ! his And , , To s to of and . have a , of for ; to in : And , : : my I , in - this I to ' ; he : d s - the , my thee
Training Batches 4473-4535: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.19s
Training Batches 4536-4598: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4598: He that will give good words to thee will flatter s , . me he have in and in not I And , To , : , and the To , I with . my ' this d the , by . his , : to of with he , ' I to have ! ? I . you , shall we I , : but the , : . you all by of . , his , , you . my : ' , be the thee s thy shall not , : not To I thy : To you of thee To and of ! of it as
Training Batches 4536-4598: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.20s
Training Batches 4599-4661: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4661: He that will give good words to thee will flatter : to I is and do , . my ? I are a ? : : her , of of ! . you be , in the . . so me the , : by to ! have , , so , To , for , To to ; , : so : To but ! thee and - as he The it , And so ' , a and of ' you my , a ! him this me in , . , . to to , , you in , , me will of him ' that d
Training Batches 4599-4661: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.23s
Training Batches 4662-4724: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4724: He that will give good words to thee will flatter thou : , and to my have with with : I all and to his , : to , of the the , : ? ? the for to to do ? . a a so my ? the - ; I And , his in , ; and ! I . . s . , , my I him , I : to is I , is s and to . shall : , of ; thou so , your ' to a - have . , that you ; you - - , I , , ' you
Training Batches 4662-4724: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.27s
Training Batches 4725-4787: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4787: He that will give good words to thee will flatter not the , this d : with ; I : to ; is and with I I , the : him . to ! ! but of will ? to he a . your of the ? ! d ! and he ! that you , And a you d ; as s I , my , , ' to my to I your : be in I s ; and . ; : ' it . ! ' is . , that he , your me . . a , ; to you shall a : : for :
Training Batches 4725-4787: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.21s
Training Batches 4788-4850: 100%|██████████| 63/63 [00:34<00:00,  1.86batch/s, loss=6.67]

Generated text at batch 4850: He that will give good words to thee will flatter . you he thee you I not the : thou ; to to : me me ' my all ! : his , , ; ; will her of my : , , it : ? and , this to And his not and I me but her your I do not a . to you : , s in I : , he your I And with of his it it that The the , : the thee . the , in , ' . , my I . of you the ' it , : ? of for
Training Batches 4788-4850: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.20s
Training Batches 4851-4913: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4913: He that will give good words to thee will flatter , will , I have . : it my so , I d , : ; And not not . , the , me : that for that so , ? to . all . the I he is d I that And , , : that , and that to do s s ' , ' , s is the d his be And . me ! , - the me : he ! in To . : do : me . thee . , ' , as . , . To all will not and have , but
Training Batches 4851-4913: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.19s
Training Batches 4914-4976: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=6.67]

Generated text at batch 4976: He that will give good words to thee will flatter thou the : , my my the his shall ? but so will ; a thee a , for have I , by be her ; - ' the and my s in . ? I . , ? , : thou , , and : the , . not the d , , the the by is I the all it so me , . be that of And to , ! his ? as but To : : I , me And , have the , you To : in in . this a shall to of .
Training Batches 4914-4976: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.67]
Epoch Loss: 6.4340, Time: 39.32s
Training Batches 4977-5039:  37%|███▋      | 23/63 [00:12<00:21,  1.85batch/s, loss=6.58]

Generated text at batch 4999: He that will give good words to thee will flatter , a - a of that : of thou the him you and s all , to his : is ! , , ' , ' , and The , ! him , . the : we a not : , me thou d thee ' to . , I s , thou : I : s the him have the we will : have , as ' I and you have , be and we : he have that me and all ' , ; ? not . , your , ' : . : him : : but
Training Batches 4977-5039:  37%|███▋      | 23/63 [00:17<00:30,  1.31batch/s, loss=6.58]
Epoch Loss: 6.4740, Time: 17.62s
</pre>


# After 5000 steps next 50 steps are run by loading the checkpoint.pth file

<pre>
using device: cuda
loaded 261973 tokens
1 epoch = 63 batches
New Vocab Size is: 19248
TransformerModel(
  (token_embeddings): Embedding(19248, 576)
  (position_embeddings): Embedding(2048, 576)
  (layers): ModuleList(
    (0-29): 30 x TransformerLayer(
      (self_attn): CausalSelfAttention(
        (c_attn): Linear(in_features=576, out_features=1728, bias=True)
        (c_proj): Linear(in_features=576, out_features=576, bias=True)
      )
      (ln_1): LayerNorm((576,), eps=1e-05, elementwise_affine=True)
      (linear1): Linear(in_features=576, out_features=1536, bias=True)
      (linear2): Linear(in_features=1536, out_features=576, bias=True)
      (norm1): LayerNorm((576,), eps=1e-05, elementwise_affine=True)
    )
  )
  (lm_head): Linear(in_features=576, out_features=19248, bias=True)
)
Total trainable parameters: 105384624
<ipython-input-11-e6e19b3533c4>:52: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
Checkpoint loaded from /kaggle/working/checkpoint.pth at batch 4999
Training Batches 4999-5061:  81%|████████  | 51/63 [00:27<00:06,  1.84batch/s, loss=6.28]

Generated text at batch 5049: He that will give good words to thee will flatter : , in . this . are , it for , - and ! for ? the have his all as ! , me we The of in : I him , . , the , And ' s so with And him . : , , of have the ' is , : : and with it of , ? , and my I of s , , ? , are , , me ' . the ' , shall . . ? : I , my my of , have that with . : and you shall I
Training Batches 4999-5061:  81%|████████  | 51/63 [00:32<00:07,  1.55batch/s, loss=6.28]
Epoch Loss: 6.4487, Time: 32.94s
</pre>