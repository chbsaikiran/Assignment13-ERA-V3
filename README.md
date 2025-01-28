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
No checkpoint found at /kaggle/working/checkpoint.pth, starting fresh training.
Training Batches 0-62: 100%|██████████| 63/63 [00:34<00:00,  1.88batch/s, loss=7.01]

Generated text at batch 62: He that will give good words to thee will flatter the , it . as . ll the the me the have ! ? in it , , shall but well ; , - if so you it , and no , , the ' the my : of be of ? d . , the the , shall ' : not , , , ! - And a the ? the and ? I ; with the the ? the , the the of ! . ' : the this . . you , ' the you ? , the have you , . , ! a she I
Training Batches 0-62: 100%|██████████| 63/63 [00:38<00:00,  1.65batch/s, loss=7.01]
Epoch Loss: 7.2789, Time: 38.12s
Training Batches 63-125: 100%|██████████| 63/63 [00:33<00:00,  1.87batch/s, loss=6.88]

Generated text at batch 125: He that will give good words to thee will flatter my : not the the I the And have we , . : the you . me is the in you PETRUCHIO it my the thy and her with , you I ' ! a ? And And I , I the you and the the our and the ' d the , be of . of , me . I . ' TRANIO : is TRANIO , the is . the our no , thy in it father not the as . thou ; say the have your . ? , the the ? a and the . ,
Training Batches 63-125: 100%|██████████| 63/63 [00:39<00:00,  1.60batch/s, loss=6.88]
Epoch Loss: 6.5660, Time: 39.48s
Training Batches 126-188: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=6.6] 

Generated text at batch 188: He that will give good words to thee will flatter and my is I with ' ISABELLA our . you of I have no LUCENTIO that all to ; To KATHARINA and you the HORTENSIO . Ay your the KATHARINA And her is the TRANIO , let of make , the The And . TRANIO is you ' s the LUCENTIO to not ' tis be is the ISABELLA : PETRUCHIO : LUCENTIO ? TRANIO the PETRUCHIO ; if the TRANIO , to To you not ' st , my the BAPTISTA . but the LUCENTIO are , I and , And the BAPTISTA for BAPTISTA . PETRUCHIO shall be
Training Batches 126-188: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=6.6]
Epoch Loss: 6.3477, Time: 39.57s
Training Batches 189-251: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=6.32]

Generated text at batch 251: He that will give good words to thee will flatter ' ll , I not , or to a a I all you be I have a KATHARINA , ' Tis . BIONDELLO - BAPTISTA : But is a the BAPTISTA with the tongue ? PETRUCHIO ? Nay : KATHARINA . PETRUCHIO , When should , here : Where with - PETRUCHIO : BAPTISTA : Signior to master I am the lord of the lord - head should not he no house the master ' ll no the world , sir is the house me ' s her her ? But no heavy not a s the world to the
Training Batches 189-251: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=6.32]
Epoch Loss: 6.0601, Time: 39.52s
Training Batches 252-314: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=6.07]

Generated text at batch 314: He that will give good words to thee will flatter , this the head . TRANIO ' d myself . PETRUCHIO ! BIANCA shall thy life ? Why thee ; to my husband the house , thou had me , the man of the state ? BIONDELLO in the house so no bed ; the father ; I ' tongue ; I do ? Come me . GRUMIO : I have be you ? PETRUCHIO : The mind ; Petruchio . My heart , my dear ? PETRUCHIO : ' tis the sun I pray , you ' tis the husband ' d ? BIONDELLO , we I may the
Training Batches 252-314: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=6.07]
Epoch Loss: 5.7985, Time: 39.51s
Training Batches 315-377: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=5.81]

Generated text at batch 377: He that will give good words to thee will flatter and no my honour , be well of your sake . HORTENSIO : No this , let me , you it so me I never you . PETRUCHIO of my lord it not not , here so , with this ; it hath you say and me to me which not . First name ? PETRUCHIO . PETRUCHIO : Ay , The duty of she might all , not be thou have you should will ? TRANIO : ' s husband in the very death ! A heart , here , all my VINCENTIO : I am that you .
Training Batches 315-377: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=5.81]
Epoch Loss: 5.5510, Time: 39.55s
Training Batches 378-440: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=5.65]

Generated text at batch 440: He that will give good words to thee will flatter . PETRUCHIO : Then , if you . TRANIO : He , not come a morning to my master , where she are and a man , sir . HORTENSIO : Why , sir , you shall not ? Why , and what have a more good wife , and a city , sir . PETRUCHIO : Bid patience . TRANIO : TRANIO : Now for him you not twenty of my husband ? HORTENSIO : Ay , he is a word . GRUMIO : Why , my wife . PETRUCHIO : He . DUKE VINCENTIO : ' s lord
Training Batches 378-440: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=5.65]
Epoch Loss: 5.3609, Time: 39.53s
Training Batches 441-503: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=5.48]

Generated text at batch 503: He that will give good words to thee will flatter . Pedant : The duke : Because , do not this . Pedant : ' s you , therefore , and love . GRUMIO : You warrant ' d be not me . KATHARINA : Now we shall you are you have him . Go will won ? LUCENTIO : Nay , you shall so you , what ? BIANCA : Ay , sir . TRANIO : My love it again ' s no master . PETRUCHIO : Why . PETRUCHIO : I will hear you ? KATHARINA : Why , as my company ? LUCENTIO : Ay . KATHARINA
Training Batches 441-503: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=5.48]
Epoch Loss: 5.1940, Time: 39.55s
Training Batches 504-566: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=5.3] 

Generated text at batch 566: He that will give good words to thee will flatter ; if I will all . TRANIO : Let me , sir : Why : I believe have a end ? LUCENTIO : And let them to be , take it not thy pleasure , for not in his love ; therefore ; And so ! I hear - morrow . KATHARINA : But to touch me to my master . ARIEL : ' d : Well , sir : There is , sir . HORTENSIO : A master . Now , where my master ! BIONDELLO : What it ? TRANIO : Now , sir with her him as
Training Batches 504-566: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=5.3]
Epoch Loss: 5.0576, Time: 39.54s
Training Batches 567-629: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=5.14]

Generated text at batch 629: He that will give good words to thee will flatter to pray . HORTENSIO : Nay , sir , sir ? KATHARINA : He is such sir ? PETRUCHIO : Let love , fair sir , I hear them good , sir ! PETRUCHIO : Pedant : Sir , sir ; I pray ! LUCENTIO : I , That you I have , And I will I take my wife , sir ? PETRUCHIO , sir ? PETRUCHIO : you . TRANIO : Pray it ! PROSPERO ! HORTENSIO : I will , I have . Provost : To bear ? KATHARINA : There ' rt you have you ?
Training Batches 567-629: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=5.14]
Epoch Loss: 4.9268, Time: 39.51s
Training Batches 630-692: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=4.96]

Generated text at batch 692: He that will give good words to thee will flatter you not a father , And thou please to - cony . PETRUCHIO : I ' d you ! I thank my brother , I did some Barnardine my horse ; he hath so take your death in a mistress at the father , you , by thy care . KATHARINA : And so mad ; no , you ? PETRUCHIO : Then see to a better ; Nor married you have ; I love to have to me ; For you are by the master ' s a little , for it , the church and by my mistress
Training Batches 630-692: 100%|██████████| 63/63 [00:39<00:00,  1.59batch/s, loss=4.96]
Epoch Loss: 4.7757, Time: 39.53s
Training Batches 693-755: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=4.84]

Generated text at batch 755: He that will give good words to thee will flatter . KATHARINA : I will do ! BIONDELLO : he , sir , he so , That my friend , you , I ' ll speak me as my father ' d , when this as I hear me a father . VINCENTIO : And I pray me : I pray me my face . SLY ' my wife . TRANIO : I do one I are with her good sir , my tongue to meet it in . PETRUCHIO : I am you have , how I ' tis a gown , sir ? PETRUCHIO : Ay , I
Training Batches 693-755: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=4.84]
Epoch Loss: 4.6622, Time: 39.16s
Training Batches 756-818: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=4.72]

Generated text at batch 818: He that will give good words to thee will flatter than my lord , I have my chamber . Pedant : I would I have have I will prove my father is to you ; have I have have a good father . PETRUCHIO : Why , That my name , sir : She be therefore I have to be a mistress : I am plain . Gremio , And I am his master : Why is a Barnardine ! TRANIO : I ' d . Was marry . TRANIO : I do her , or do I have marry , and therefore my master : It . KATHARINA :
Training Batches 756-818: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=4.72]
Epoch Loss: 4.5564, Time: 38.97s
Training Batches 819-881: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=4.56]

Generated text at batch 881: He that will give good words to thee will flatter . PETRUCHIO : Sir , sir : I may her , we must not you : TRANIO : I would not do no beast here , For I beseech you . PETRUCHIO : Sir , if he warrant her my father ! PETRUCHIO : O she have frown against her . PETRUCHIO : I do you must not , And by such the matter . I ' I have not this to see , Hortensio , thou have I beseech the first of her of her ten . PETRUCHIO : Marry , if Kate ! Pedant : Then ' d
Training Batches 819-881: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=4.56]
Epoch Loss: 4.4514, Time: 38.96s
Training Batches 882-944: 100%|██████████| 63/63 [00:33<00:00,  1.87batch/s, loss=4.44]

Generated text at batch 944: He that will give good words to thee will flatter my first , And she ' ll therefore no finder her ? Angelo , will be be a priest and half than the name as you , As left , if a mad , with , he knew the other a dozen blood . GRUMIO : I could , that if I speak . VINCENTIO : if that I ' twas a good , were you will , if she is good the manner are the poor in her women for the king and a hundred of her , a dozen women was , That I know ' s love
Training Batches 882-944: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=4.44]
Epoch Loss: 4.3508, Time: 39.08s
Training Batches 945-1007: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=4.41]

Generated text at batch 1007: He that will give good words to thee will flatter him he hath you , Which with my father you , That you ; And yet thus offer you will I be answer me . He is , sir , if you , if he comes you ; For me , I love you ? GREMIO : In private . DUKE VINCENTIO : if you my son ! that I fear , do him wrong , to see me . LUCENTIO : You will be forward . LUCENTIO : And a thing ? BAPTISTA : Why wrong ? CORIOLANUS : How he , he , but , As , one
Training Batches 945-1007: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=4.41]
Epoch Loss: 4.2432, Time: 38.95s
Training Batches 1008-1070: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=4.26]

Generated text at batch 1070: He that will give good words to thee will flatter her my king , if you , If we shall I know , which I thank you . GREMIO : I would say here will confess to it not ? PETRUCHIO : To Rome , I know the king ! Gremio ! PETRUCHIO : BUCKINGHAM : Gentlemen ; And let him ; And be too my lord , good lord , and I do it seems a masters - morrow ; Because my mistress ; And , That I ' d . TRANIO : If my lord ; And who came it , My lord ? HORTENSIO : we ,
Training Batches 1008-1070: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=4.26]
Epoch Loss: 4.1603, Time: 39.09s
Training Batches 1071-1133: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=4.15]

Generated text at batch 1133: He that will give good words to thee will flatter ? What is her himself ; for the king with shame , that may be have the king do , and a little , To say you you ? who please ; The villain ' s death . VINCENTIO : The man ! he ' ll say , where he seems ? TRANIO : Now , being the old ? First Citizen : and the time , The king ; so , he hath said ? My good I would have a man ! I hear , a wise , and , what , good , a man ' me
Training Batches 1071-1133: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=4.15]
Epoch Loss: 4.0614, Time: 38.88s
Training Batches 1134-1196: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=4.04]

Generated text at batch 1196: He that will give good words to thee will flatter him To make him . DUKE VINCENTIO : But he he does his father , Kate , For his fault , Or so , I , to bed ? Come it so as her love , for hand , Which if well his son , that to think your very first son , if thy father from the moon , Whose father ! What makes him to do me to your father beseech you . First Citizen : What you live ? CLARENCE : But I do you not to the case ' Fore me ? BAPTISTA : I will
Training Batches 1134-1196: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=4.04]
Epoch Loss: 3.9632, Time: 38.94s
Training Batches 1197-1259: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.93]

Generated text at batch 1259: He that will give good words to thee will flatter us , I pray you . PETRUCHIO : What , she ' s child . TRANIO : GREMIO : Well ? DUKE VINCENTIO : Ay , let me , sir , sir . PETRUCHIO : The tale let me leave him . MENENIUS : Yes , and it the duke is not you both the people . PETRUCHIO : Well , if you do it is ? a word , sir ; let me speak to her good merry and one woman , what fellow , And by the ice , If he ? TRANIO : Who was a gentleman
Training Batches 1197-1259: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=3.93]
Epoch Loss: 3.8668, Time: 39.00s
Training Batches 1260-1322: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=3.86]

Generated text at batch 1322: He that will give good words to thee will flatter than I am so , And let me , to ' d , Have it thus -- All cunning . LUCENTIO : The will be won on you these not my father , for me . KATHARINA : For private , my lord , what is your name , Never now , I am my power ? The boy , Gremio , I beseech you are of heaven ? For one of my countenance to get him , If the law ? DUKE VINCENTIO : The favour are now I do it , I am mad before , I must
Training Batches 1260-1322: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=3.86]
Epoch Loss: 3.7751, Time: 38.95s
Training Batches 1323-1385: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.77]

Generated text at batch 1385: He that will give good words to thee will flatter him by time ? GRUMIO : This is wrong ? what you , and I am content or else to him : But , I think you might do be first so : O , and I beseech you . What , you ; He hath ; If it , yet , If you , sir , And , And hear the prison with my servant , Were But I am a jot from me a I am content to use you truly , If I ' d , to your patience to use is he knew for me this
Training Batches 1323-1385: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=3.77]
Epoch Loss: 3.7055, Time: 38.99s
Training Batches 1386-1448: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=3.68]

Generated text at batch 1448: He that will give good words to thee will flatter it with a father ? For you to his father ? ESCALUS : KATHARINA : And that Petruchio ! I love he do a father ? if not his news have all a man , To love indeed her no thought you are to ' tis known , I pray thee , or you , but a confirmed love , Let me do it come about a silken , And give him in him do not to me ; and therefore a shrew of private to his face . TRANIO : I am a cup for my father ' t
Training Batches 1386-1448: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.68]
Epoch Loss: 3.6343, Time: 39.08s
Training Batches 1449-1511: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.61]

Generated text at batch 1511: He that will give good words to thee will flatter As his quarry As you . PETRUCHIO : Good time you I thank you are no more , I love ; And so did rather therefore he Minola , I am so wrong , where I beseech you love yet what I will not to you so I am out . PETRUCHIO : To do you show this wrong you are obedient in such to make me , I beseech you were as To meet upon me in this hand , Where I do beseech the city like patience To the choice ; But to make your honour ,-- TRANIO
Training Batches 1449-1511: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.61]
Epoch Loss: 3.5629, Time: 39.06s
Training Batches 1512-1574: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=3.53]

Generated text at batch 1574: He that will give good words to thee will flatter him his friends is so . I have , Which I have the old in a other lack - rule where I do I would have you can the people that you show me in : That by them do it . TRANIO : I am provided out ; Never never be , and so , so that that I show me , And to say . Is out to the hand , for no fit you . The charity ? What can I must have , and these man of good he is it hath , and as almost
Training Batches 1512-1574: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=3.53]
Epoch Loss: 3.4945, Time: 39.00s
Training Batches 1575-1637: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.5] 

Generated text at batch 1637: He that will give good words to thee will flatter him in time . DUKE VINCENTIO : This is you do I am one within thy husband , for request ; or where he knew no more than an Murderer : O that is a man live in a tabour of his respected fair . What , Which would not live might he hath so woo not access and carried live : To do not to him , The mad of a wrong ; But to serve , be not to be barr ' s be gone . KATHARINA : And then he did not what he shall not so
Training Batches 1575-1637: 100%|██████████| 63/63 [00:39<00:00,  1.62batch/s, loss=3.5]
Epoch Loss: 3.4373, Time: 39.00s
Training Batches 1638-1700: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.44]

Generated text at batch 1700: He that will give good words to thee will flatter me ? BAPTISTA : What him for his brother , Who now To you I beseech you ? VINCENTIO : See by the thing as yet mock him to go so much ; and withal , and yet so , Let , by your sister on him by that he hath hath he hath so so hath not possible with her . DUKE VINCENTIO : Who is out by his bed : In false a thing ? To think ' s to live ; And by nature came The one . DUKE VINCENTIO : what might be a man ,
Training Batches 1638-1700: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=3.44]
Epoch Loss: 3.3843, Time: 38.97s
Training Batches 1701-1763: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.36]

Generated text at batch 1763: He that will give good words to thee will flatter me . PETRUCHIO : I had an you not to be wholesome her as you , who would please ; And never die , And may show ' d withal , And , In what , GREMIO : I have done . GREMIO : the thing of this is so , he bids me for my fair sweet gentleman may be as she comes so , how himself . Why , ' d , sir , And he will I will be married ? GREMIO : Indeed , To know that , That had rather bid me to see me
Training Batches 1701-1763: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.36]
Epoch Loss: 3.3296, Time: 39.06s
Training Batches 1764-1826: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=3.28]

Generated text at batch 1826: He that will give good words to thee will flatter you ? TRANIO : I , and let me at your own , And then , And bear to bear a time , If that may scorn The traitor , The neck ; And to die to rejoice his neck -- BIONDELLO : Let me in absence thou not suffer VINCENTIO : To do ' d for virtuous of all law , I do beseech you -- TRANIO : My lord , Mark not before thee , To this news from him , I had I hear , And therefore , With my best ? TRANIO : And to die
Training Batches 1764-1826: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.28]
Epoch Loss: 3.2779, Time: 39.14s
Training Batches 1827-1889: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.24]

Generated text at batch 1889: He that will give good words to thee will flatter me . TRANIO : If you sir , sir , he hath given , Was he were thus ? O a deputy will be ? TRANIO : There ' en his sake . ESCALUS : Call some man ; And he is when you , is his great age shall live , that the deputy is born : I must be found him And bear him . DUKE VINCENTIO : He will be he when he , And , and my horse . DUKE VINCENTIO : An he hath his honour , how she has the first to him to
Training Batches 1827-1889: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=3.24]
Epoch Loss: 3.2240, Time: 39.00s
Training Batches 1890-1952: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.23]

Generated text at batch 1952: He that will give good words to thee will flatter him . ESCALUS : Let me this , let you will not , he seems , ere it to me what you ? ESCALUS : For will not the provost . DUKE VINCENTIO : Ay , Petruchio ! DUKE VINCENTIO : That hath some thing That offer than the Volsces would beseech you are his love and so good sir . DUKE VINCENTIO : I knew this his brother may have heard his device ; Who will not . DUKE VINCENTIO : And , And will , for their hand , For it ? TRANIO : Why , that one
Training Batches 1890-1952: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=3.23]
Epoch Loss: 3.1842, Time: 39.00s
Training Batches 1953-2015: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3.2] 

Generated text at batch 2015: He that will give good words to thee will flatter of I lived to be what , And new times , but a manner and his country you on his death , The father , For one , Or thus we promised so in arms affords him not by his country : Alas , For with him , Let his best : He is wise . Now , nor Elbow : What , where now , And : Go ? VINCENTIO : Fair Gentleman : Yet , what will call a word , noble duke be your noble heart be speak you ? PETRUCHIO : O new ? why ,
Training Batches 1953-2015: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=3.2]
Epoch Loss: 3.1540, Time: 38.99s
Training Batches 2016-2078: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=3.16]

Generated text at batch 2078: He that will give good words to thee will flatter me That I am one . DUKE VINCENTIO : What you not not , That he should not to this men at such half what these thus : It is so thing . TRANIO : I beseech you are ? KATHARINA : yet to hear your noble as a man . PETRUCHIO : you be it so much more than I pray , I beseech you . BAPTISTA : I beseech you , I know my master Minola ? KATHARINA : What to you to crave you please you do her good , That ? TRANIO : The gentleman of
Training Batches 2016-2078: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.16]
Epoch Loss: 3.1159, Time: 39.15s
Training Batches 2079-2141: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=3.1] 

Generated text at batch 2141: He that will give good words to thee will flatter ? I saw me were a worthy to the neck . DUKE VINCENTIO : I have given to be born to , then since you . GREMIO : Away did you know not the court . PETRUCHIO : Nay , sir , no more I think you well , I pray you are you come to meet ? PETRUCHIO : But this one seems to this strange . PETRUCHIO : How therefore thus ? poking , sir , how you your honour , if you , you are no ? KATHARINA : Ay , if it is it ' ll
Training Batches 2079-2141: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.1]
Epoch Loss: 3.0808, Time: 39.04s
Training Batches 2142-2204: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=3.06]

Generated text at batch 2204: He that will give good words to thee will flatter men ; Commend me out of white and evil , If I know the court and humbly for true own and his cause Of execute her unhappy ; but lay from duty may not dark : There had , With all the law , who ; No . FLORIZEL : if I am come on them good with you have as you read with the more . GREMIO : Or else ! Worthy Barnardine may break deeds , were ? TRANIO : Be those that I see : He shall please you with what it too . LUCENTIO : No
Training Batches 2142-2204: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.06]
Epoch Loss: 3.0435, Time: 39.13s
Training Batches 2205-2267: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=3.06]

Generated text at batch 2267: He that will give good words to thee will flatter her That dared to take it . O tears to do , Is one enough , if you hold tears with private ; and honour ' d me a chamber ; Another seven , Yet so I do I am no more , For the very advice of the court : if you all . O you , I had die the very house , if you saw me , have not speak to be on you swear to be so , though you love his father can find us , Signior Gremio : Do you . ESCALUS : But
Training Batches 2205-2267: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.06]
Epoch Loss: 3.0246, Time: 39.13s
Training Batches 2268-2330: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=3.02]

Generated text at batch 2330: He that will give good words to thee will flatter at a white - draw on for my first ? If you do one as man : Why , I thank you ; I humbly not show , I beseech it possible , for oath , for my fortunes to get again , or it . GREMIO : Why , I can I think or fall or he comes my heart to the thing , To you be he ' d : Be hatch ' ll show me on I think you show me a fool please a great honour of joy And therefore ; He ' er all the
Training Batches 2268-2330: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3.02]
Epoch Loss: 3.0000, Time: 39.21s
Training Batches 2331-2393: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=3]   

Generated text at batch 2393: He that will give good words to thee will flatter snow of mercy . FRIAR PETER : I were needful given , sir . Provost : I had mine fellow , help me from me to bear his face . What you were a man ; I do to come thus by him ? I am none shall not possible ; for it not that new - Servant : he that long for a man ? TRANIO : By his foot than I can do to do it is a plague I thank your mistress ' s honour , let ' ll know a poor duke comes he no wit
Training Batches 2331-2393: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=3]
Epoch Loss: 2.9748, Time: 39.01s
Training Batches 2394-2456: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.99]

Generated text at batch 2456: He that will give good words to thee will flatter you will do a gentlewoman ? DUKE VINCENTIO : I call him do , I love I know his favour . ESCALUS : Ay , and bad . POLIXENES : Gentle my lord , my lord ; for him to forswear him be your chamber Is thou hast burnt ; Nor ' d him so well I beseech you ere this ? DUKE VINCENTIO : How ? TRANIO : By heaven , sir , ha : I love no ; nay , you have sorry think how much we make one licence and poking , But , I can be
Training Batches 2394-2456: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.99]
Epoch Loss: 2.9458, Time: 39.09s
Training Batches 2457-2519: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=2.96]

Generated text at batch 2519: He that will give good words to thee will flatter . DUKE VINCENTIO : He did this nigh , that scorn . DUKE VINCENTIO : He is a woman on my neck ,-- DUKE VINCENTIO : Dispatch . Why that he , sir , ha : The gracious good my father , I fear longer . DUKE VINCENTIO : Now know no gentleman ! PETRUCHIO : He is not to be time to hear me to him to his honour ! I know her he must have , gentle that honest he hangs on a means himself when he were he seems to accuse by that I beseech he did
Training Batches 2457-2519: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.96]
Epoch Loss: 2.9218, Time: 39.04s
Training Batches 2520-2582: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.97]

Generated text at batch 2582: He that will give good words to thee will flatter heaven , And so thus he is her friends to hear somewhat As one flourish Of one thing Than this heaven , or born court ' tis , Or therefore to me himself , which late to do do not mad sense even that , Let heaven , ha , Who , if this afternoon , if he hath no power for you to a dangerous gentleman , do not those of my mind , And born , And know the other person and think is no longer night . For I have ; therefore been mad of my wife
Training Batches 2520-2582: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.97]
Epoch Loss: 2.9166, Time: 39.12s
Training Batches 2583-2645: 100%|██████████| 63/63 [00:34<00:00,  1.86batch/s, loss=2.94]

Generated text at batch 2645: He that will give good words to thee will flatter , I will not scorn again in heaven . I have meet you had find - VI VINCENTIO : Sir , to me to do not make her lord , That you well in a farmer came to do not my life to heaven . PETRUCHIO : I hear : Why , he hath mine ? ISABELLA : And I know my question your honour ; look ' t for this . I think you , Petruchio ' t , that I love a ignorant allied ' t to me , you know me ; he deserved the word Shall
Training Batches 2583-2645: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.94]
Epoch Loss: 2.8999, Time: 39.01s
Training Batches 2646-2708: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.92]

Generated text at batch 2708: He that will give good words to thee will flatter and I had him straight to heaven to night . DUKE VINCENTIO : DUKE EDWARD IV : I do , what to a subject ' tis burnt for this is a gentlewoman . GREMIO : I prithee hear him . What , how you speak ? what is the deed is no power ? What ; ha ? O mistress ? Speak to offend her ; I profess ; the duke ' s my lord , and Katharina , he be a very very gentleman , ha ? I cannot crave her cousin , what the duke ? O ,
Training Batches 2646-2708: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.92]
Epoch Loss: 2.8810, Time: 39.06s
Training Batches 2709-2771: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=2.89]

Generated text at batch 2771: He that will give good words to thee will flatter of it will perceive What a life : I won I have the king perform no better to do a name . I am , if this scolding love my faith , I am dared in this court or have been : I know ' erwhelm you shall hear you were out of love ; If I pray you love ? DUKE VINCENTIO : He can hit her , If not , marry me to be ta ' ld go , To do me not , And I mean Were he , Which I am given to meet me ,
Training Batches 2709-2771: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=2.89]
Epoch Loss: 2.8610, Time: 38.99s
Training Batches 2772-2834: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=2.86]

Generated text at batch 2834: He that will give good words to thee will flatter him out of child , That his face ' d on the one , And take their request had been but by his father and therefore to be one shall not so court of my friends so Angelo , or one hath kept before . Second Gentleman : if he hath need but this . DUKE VINCENTIO : The child ! he is the devil , I hear on the truth of my father not so , I must be great women are very strange and though we have not ; let us here , For he is that favour
Training Batches 2772-2834: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.86]
Epoch Loss: 2.8416, Time: 39.06s
Training Batches 2835-2897: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=2.85]

Generated text at batch 2897: He that will give good words to thee will flatter him To scorn him , That the one flourish Which , But when he not suffer ' d so ; For I humbly beseech you ; you might think I do beseech you like a king , you take your father : Why , If it ? By your friend , Of what you be such my sweet father love The queen to do it ? what he that I ; For Gloucester ; And did die , Who will be not on me ? That , I beseech you should speak , To stand that will not on me
Training Batches 2835-2897: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=2.85]
Epoch Loss: 2.8258, Time: 38.98s
Training Batches 2898-2960: 100%|██████████| 63/63 [00:33<00:00,  1.86batch/s, loss=2.83]

Generated text at batch 2960: He that will give good words to thee will flatter me against my lord , to his power or I dare not . KING RICHARD III : I beseech you know , Or , I will , so : Yet expect to make you no part : So yours . LORD selves : he should be what long ? First Citizen : Very the queen from either him dead ; beseech you ? are well to the contrary . KING HENRY VI : My lord , you know you are so ? CLARENCE : Well a sin . HERMIONE : What do not , When dear he , I think
Training Batches 2898-2960: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.83]
Epoch Loss: 2.8014, Time: 39.15s
Training Batches 2961-3023: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.81]

Generated text at batch 3023: He that will give good words to thee will flatter adversaries ! DUKE VINCENTIO : By thine , I mean thy daughter for soon abused To keep your bed , I saw one Have well for what ' t are able to - one ere when I ' s point . FRIAR LAURENCE : I pray this Prince of you , Or else please my wife , Nor told her As I ' t , To scorn ' tis known ; which I , he did bear , he spake thy man , O monstrous , he does see this hand , to my lord , I have a woman
Training Batches 2961-3023: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.81]
Epoch Loss: 2.7789, Time: 39.03s
Training Batches 3024-3086: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.79]

Generated text at batch 3086: He that will give good words to thee will flatter men . O lady ! DUKE EDWARD IV : He can no more thousand and me and her good blood all with us . DUCHESS OF YORK : DUKE VINCENTIO : O then , and here full as I will be so , who shall ; O , for I thank thee , Prince IV : My sovereign ; I am called to say ; for no more . KING HENRY VI : Who , I am a wise . DUKE VINCENTIO : Ay , belike they may think ' s sake . KING RICHARD III : Be thou ?
Training Batches 3024-3086: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.79]
Epoch Loss: 2.7593, Time: 39.06s
Training Batches 3087-3149: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=2.77]

Generated text at batch 3149: He that will give good words to thee will flatter courtesy , and myself , But so on aught where he was , I will do perform to the court Of what he was mad men of charity , I will . I would say , who I understand ? BUCKINGHAM : I thank me not grief of such a father , I will be born to the law to do beseech you here , to me . LADY art ; and he ; And you do , if I do I beseech you to me , for it . GREMIO : I do beseech you ; and I never
Training Batches 3087-3149: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.77]
Epoch Loss: 2.7415, Time: 39.16s
Training Batches 3150-3212: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.76]

Generated text at batch 3212: He that will give good words to thee will flatter him . QUEEN ELIZABETH : Is that you not certain more gentleman of love . DUKE VINCENTIO : And hear , That he ? DUKE OF CLIFFORD : No , no other this hangs on my master withal , a boy ; a score , I know ' t ' s true , I pray you love the subject to hear you from it comes to thee , I think thou so did knowledge to what , to the field . HORTENSIO : The man , beseech you of us at the death . O , I cannot ' t
Training Batches 3150-3212: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.76]
Epoch Loss: 2.7235, Time: 39.10s
Training Batches 3213-3275: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.74]

Generated text at batch 3275: He that will give good words to thee will flatter happiness ; and his lord , That I won . KING HENRY VI : I beseech you at no , I am not that so ? Romeo ' s dead than the matter , ' d : When I know the very time ; In his ear as by torture , make us , no , was long we will not no issue of redress of him to say , do honour , the virtues to him . HENRY BOLINGBROKE : Good hand , the tears , thou this gentleman thou think it ? DUKE VINCENTIO : Good thee a
Training Batches 3213-3275: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.74]
Epoch Loss: 2.7108, Time: 39.12s
Training Batches 3276-3338: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.73]

Generated text at batch 3338: He that will give good words to thee will flatter - All time , What his love . Servant : What I think now I reckon and so great brother ' t I have beseech you ; and a villain , what he hath speeded at love I am no hope ? I never spake ; what I dare , Which , you follow him from my love , I must bear the butchers of York . I think my brother than wrong no need to the devil That he at some dream ; Who speaks it on the man should come to joy , It was done . DUKE
Training Batches 3276-3338: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.73]
Epoch Loss: 2.6989, Time: 39.03s
Training Batches 3339-3401: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.72]

Generated text at batch 3401: He that will give good words to thee will flatter me a worser the time hath known . KING RICHARD : He will pluck him too foot no issue : Pray ' er ' s some other ; let the shepherd , and flourish upon your name beseech you pardon her I , I say , come on name for what , Shall I am so too late : say , How else not , I hear out of your honour . I am show ' d . RICHMOND : Sir , When you hit it is your name to crave the prayer : Lord ELIZABETH : The vile ,
Training Batches 3339-3401: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.72]
Epoch Loss: 2.6879, Time: 39.06s
Training Batches 3402-3464: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.71]

Generated text at batch 3464: He that will give good words to thee will flatter you When tears ? Have you know this lean are born . DUCHESS OF YORK : See thou beseech the duke , am one noble suit , I have out of such a villain ' d his hand , Which for mine . I was no cause that I must not , thou mayest VINCENTIO : One , not to this gentleman . I will make it be called me , ' d With thine lap of love , When dying than his own groans : No man ; Which take out of them , Or thou , When scarce
Training Batches 3402-3464: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.71]
Epoch Loss: 2.6774, Time: 39.04s
Training Batches 3465-3527: 100%|██████████| 63/63 [00:34<00:00,  1.86batch/s, loss=2.7] 

Generated text at batch 3527: He that will give good words to thee will flatter courtesy . DUKE VINCENTIO : And about thee to this proud gentlewoman , who , I thank your tale to - bear ; who : for him , I know thee a forsworn , There shall not hear me to hear and therefore are silent , The very name out : By me , no , What , Because I ' s face . KING RICHARD III : I will teach me . TRANIO : let her , sir ; and if I think ' s name , for any wrong . DUKE VINCENTIO : Sweet that is still him
Training Batches 3465-3527: 100%|██████████| 63/63 [00:39<00:00,  1.62batch/s, loss=2.7]
Epoch Loss: 2.6653, Time: 39.01s
Training Batches 3528-3590: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.69]

Generated text at batch 3590: He that will give good words to thee will flatter courtesy , what I will no doubt . QUEEN ELIZABETH : Know you , sir ? PETRUCHIO : Who shall , I have upon your mistress , he Grey , I must I profess . LEONTES : A simple presence , to make me , he , if you have wilt have hate me , Which I will not his mistress ; For time there ; Sir , I beseech you not , So said behind us to her . GLOUCESTER : Sir , Which I shall I , beseech you go and yours in the new name it :
Training Batches 3528-3590: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.69]
Epoch Loss: 2.6582, Time: 39.06s
Training Batches 3591-3653: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=2.68]

Generated text at batch 3653: He that will give good words to thee will flatter me , I will not a love . KING HENRY BOLINGBROKE : Why , then . FLORIZEL : You tell me charged you , I beseech you have no manner I am known done . ESCALUS : What bed , tell me have her to her tale that we will bear : No , what news to such heart ? DUKE VINCENTIO : I do you , to crave his mistress . ESCALUS : Then ' s to hear me , sir , if he the ground , madam , thou art he , that ' s mild to have
Training Batches 3591-3653: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.68]
Epoch Loss: 2.6515, Time: 39.18s
Training Batches 3654-3716: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.68]

Generated text at batch 3716: He that will give good words to thee will flatter adversaries , which God not so , I will wet him . LEONTES : What , what no more ; I am possible ; I will you not the king I will put The wrong ; only that I speak to bed - disquiet : He did , and I the soul of men have known . DUKE VINCENTIO : What , sir : What witness when you ; I do you , good man ? DUKE VINCENTIO : the duke , where an such a mistress ; And he did offer , I should bear it is I swear
Training Batches 3654-3716: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.68]
Epoch Loss: 2.6451, Time: 39.07s
Training Batches 3717-3779: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.67]

Generated text at batch 3779: He that will give good words to thee will flatter adversaries . DUKE VINCENTIO : The plain men were given , when I must have beseech him to her a worthy one thing . Second PETRUCHIO : Well , father ! SICINIUS : If you must not hear me . DUKE OF ELY : Indeed , Let her but , that did long , to this abused here in their five matters with him by the prince , I mean to Rome , And when I thought than a thief , as I thank you , We will pluck her to command here . Hark , to her my power
Training Batches 3717-3779: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.67]
Epoch Loss: 2.6387, Time: 39.07s
Training Batches 3780-3842: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.66]

Generated text at batch 3842: He that will give good words to thee will flatter heaven withal I do not for Juno than thou ; Or I am even as done , The state Is his children ; if though I will be a wrong thee : Be married of her that ever I love them . GLOUCESTER : I have , I would he was wrong , my request me , Or I will not her true I go to me , Than love . I do , I will time , do me , I , so long . MENENIUS : It is did I will keep it , that is good good
Training Batches 3780-3842: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.66]
Epoch Loss: 2.6308, Time: 39.10s
Training Batches 3843-3905: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.66]

Generated text at batch 3905: He that will give good words to thee will flatter - fast and execute him . DUKE VINCENTIO : There shall be so , I might live to beseech heaven . DUKE VINCENTIO : He enjoys him so an time I have given To love . MARIANA : He Grey ! PETRUCHIO : Why can thy true fair cousin you ? what is my good ; If , he did I am not : Since I shall be so I have her a father out of them , his son . But see her a man ? But come from me swear and leads . POLIXENES : What good sir
Training Batches 3843-3905: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.66]
Epoch Loss: 2.6265, Time: 39.04s
Training Batches 3906-3968: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.65]

Generated text at batch 3968: He that will give good words to thee will flatter heaven ? DUKE LAURENCE : It will be so , or a month , I beseech thee , I have , I do he can humbly learn him . Why the prince , for I , he ; For not this dead ,' he would beseech so be I speak , For left no man , if he had I live With mild soul : And the Lord : He was the man have me of honour , I ' d his honour , and honour , do it . KATHARINA : Let me be my good , for one
Training Batches 3906-3968: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.65]
Epoch Loss: 2.6224, Time: 39.07s
Training Batches 3969-4031: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.65]

Generated text at batch 4031: He that will give good words to thee will flatter earth . Now , If the gentlewoman , then you do , my charge me , here , I have mine looks ' d upon , I thank thee , my father . DUCHESS OF GAUNT : But what thou know , ' st I love him how so ? I , good young father , As joyful father , thy state ; so there was a mistress ! If the dead , unwilling oratory That him live with thy sovereign , though I speak with his grace .' Did all I do dead , Thy brother was his face
Training Batches 3969-4031: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.65]
Epoch Loss: 2.6183, Time: 39.03s
Training Batches 4032-4094: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.65]

Generated text at batch 4094: He that will give good words to thee will flatter men . O tears , cousin , to Claudio , he go on , I am satisfied . KING RICHARD II : The child ' s banishment ; and Harry A man that he was , he had set the tale to the neck of your own than to forswear him , if I speak ; And thus ? RICHARD III : What beseech you love there , it ? DUKE OF AUMERLE : A worthy he did known too much to the great - morrow : Why , I should not to yours for him . KING RICHARD III
Training Batches 4032-4094: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.65]
Epoch Loss: 2.6143, Time: 39.06s
Training Batches 4095-4157: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.64]

Generated text at batch 4157: He that will give good words to thee will flatter ; I did do not myself , therefore tied , O lady . GLOUCESTER : Away to be my honour ' d for their heart ; and thy father ; I tell you , I was a king : I have I wish to be done me . What e ' s : No , will be given away ; I am no man : I swear : I know to thee . PETRUCHIO : You will be full of you so much , I will not have been in this rude sooner yours , the sound , I must
Training Batches 4095-4157: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.64]
Epoch Loss: 2.6091, Time: 39.04s
Training Batches 4158-4220: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.64]

Generated text at batch 4220: He that will give good words to thee will flatter heaven ? What ' s a pleasure ; and seven one when you , Is love , dying thing not not yet . O my joints doth thy neck , then , be so late , who cannot see what thou thyself , to see thee , there is now , and services on her . KATHARINA : By my poor word , I beseech you To be long he should to my mistress ; and take them on a wise with God call him a man of ours . GRUMIO : Ay , Or so much before the one
Training Batches 4158-4220: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.64]
Epoch Loss: 2.6064, Time: 39.15s
Training Batches 4221-4283: 100%|██████████| 63/63 [00:34<00:00,  1.86batch/s, loss=2.64]

Generated text at batch 4283: He that will give good words to thee will flatter me but what mistress , To the neck ere I beseech you . DUKE OF YORK : I beseech you your father , When true , I pray you show : I ; you beseech you the duke ; And durst keep his bed . DUKE VINCENTIO : Then I love , Or be obedient an merry . STANLEY : Come on the field i ' t to the present answer her a word . I beseech you so how , That is that - proud to a woeful presence , Or whether you are very true , Who should
Training Batches 4221-4283: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.64]
Epoch Loss: 2.6038, Time: 39.10s
Training Batches 4284-4346: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.63]

Generated text at batch 4346: He that will give good words to thee will flatter . What love me on ? I mean possible ? Keep scene of a faults ? My child : All : I hate , When you in his faces ; And , and it ? I mistake so embrace upon thee , Let thee , for this deed : Yes , my son , so good man too much from thy quarrel with the king cannot , ay . Pray ! Why know I am silent with thee ; so young woman are thou varlet Has not live . His voices ! After , a merry . Yet yet thou
Training Batches 4284-4346: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.63]
Epoch Loss: 2.6011, Time: 39.07s
Training Batches 4347-4409: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.63]

Generated text at batch 4409: He that will give good words to thee will flatter afford at a law , When it and virtuous sorrow , I was no one And now I do ; And call him Than be love : the court is yet not lodged or to be a great , Which one that hath seen ; for I beseech you can bear : I hear her , I had , and so pent friends . KING RICHARD II : We have , hear me ? Sir , sir , you well my sorrow ; if you speak ; And sure on mine ? where : if yours that I ' d
Training Batches 4347-4409: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.63]
Epoch Loss: 2.5985, Time: 39.10s
Training Batches 4410-4472: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.63]

Generated text at batch 4472: He that will give good words to thee will flatter happiness withal o ' s time ! But her two , I thank you your hand Which had used than the king no less Than not have ; the ice , I hold your lineaments : I do beseech you to her this youth of your name of state in them , which of proud , I know me here to be deliver ' d , ' t , And being so kiss ' Tis in question . Lord Should die . Is death to my pity me ; for his honour : I do me on my mistress !
Training Batches 4410-4472: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.63]
Epoch Loss: 2.5951, Time: 39.11s
Training Batches 4473-4535: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.63]

Generated text at batch 4535: He that will give good words to thee will flatter me A white eyes , And this night , I fear to him To the bark upon so , By what I pray him ' Tis given , Is when any name . QUEEN ELIZABETH : I know no name you in the noble men of heaven , love hath burnt with me , The very old tale : And I beseech you before thy need . This love who , that shall not a glass , or I , When forsworn again , love you , Or much love ; for I would please the court of the matter
Training Batches 4473-4535: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.63]
Epoch Loss: 2.5933, Time: 39.05s
Training Batches 4536-4598: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.62]

Generated text at batch 4598: He that will give good words to thee will flatter name ? DUKE LAURENCE : what I humbly let me , There is silent years to be measures . BRUTUS : That if I warrant no saint , That I ' d Claudio , and in death Till you no other Has never can born . DUCHESS VINCENTIO : No , One word : The wanton thee so no ; And he am gone , I said ' s son ! where he had been gone on my mother , is not by his saint , who is his majesty , But by the selfsame justice ; Be thou please
Training Batches 4536-4598: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.62]
Epoch Loss: 2.5916, Time: 39.06s
Training Batches 4599-4661: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.62]

Generated text at batch 4661: He that will give good words to thee will flatter heaven afford and many cony , And who comes their sorrow please where sorrow , is All : Why , the king ' s death ? If you may not so : The king , father , What , I thank you not be so I love ? why not be sold , Even to her it hath not on ' s sound ? When I would be so much As I think so so did , That there : He hath so his name - breach ELIZABETH : Thou liest , As he pluck that all and his presence
Training Batches 4599-4661: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.62]
Epoch Loss: 2.5899, Time: 39.05s
Training Batches 4662-4724: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.62]

Generated text at batch 4724: He that will give good words to thee will flatter reproof , And you pluck them on , Or you with safety , may accept you were even to him ready to his dancing strength from his opinion , We shall follow me on it . HENRY BOLINGBROKE : Who , to lose a thing but I am provided that comes ? What forbid ? I am thus to this name ; And you have , consider With our power ; And shall you well of him to answer you , And he comes too , who will to bear you deliver ' d To understand , and to live
Training Batches 4662-4724: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.62]
Epoch Loss: 2.5882, Time: 39.15s
Training Batches 4725-4787: 100%|██████████| 63/63 [00:33<00:00,  1.85batch/s, loss=2.62]

Generated text at batch 4787: He that will give good words to thee will flatter noses , And know the king ere it not so be woman of state ; And so , and I thank them to death to ; no particular , When thou sold her but my children , He owed in heaven , What , being a king . What says so respected court York , will not so bastard ? By this great hand : Is noble lord , I cannot not for me ? I am , That hath thy name nor great - love That he were himself , Nor never brought there comes he were to wail
Training Batches 4725-4787: 100%|██████████| 63/63 [00:38<00:00,  1.62batch/s, loss=2.62]
Epoch Loss: 2.5859, Time: 38.96s
Training Batches 4788-4850: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.62]

Generated text at batch 4850: He that will give good words to thee will flatter them in justice - point of it . DUCHESS OF cousin age , When I have to give me this ; All : I think , which and scorn ? DUCHESS OF YORK : Why that they do beseech you , to your ladyship , since from London , That Edward , my love was not to the field ; and thrive to what ' twere a woman ; The league of patience . KING RICHARD II : but I cannot not are I know myself ? Why should do to die . DUKE VINCENTIO : What , my grace
Training Batches 4788-4850: 100%|██████████| 63/63 [00:39<00:00,  1.62batch/s, loss=2.62]
Epoch Loss: 2.5848, Time: 39.01s
Training Batches 4851-4913: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.62]

Generated text at batch 4913: He that will give good words to thee will flatter me to the air , For my state , If you I had , who lineaments , good boy to you mistake his bidding . Take him not myself ? O , sir : What will you your presence ! ESCALUS : Thanks , I am love ? LEONTES : Indeed , I am your noble lords . HASTINGS : beseech you as you say ? Lady commends ? what ; what I beseech you love Which , I do so else he either a man ? TRANIO : I will , As truth betwixt him , when I trust
Training Batches 4851-4913: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.62]
Epoch Loss: 2.5836, Time: 39.07s
Training Batches 4914-4976: 100%|██████████| 63/63 [00:34<00:00,  1.85batch/s, loss=2.61]

Generated text at batch 4976: He that will give good words to thee will flatter death ! What love from Rome , There ' s well boy , where thy boon , I am seems ,' I send That I was past , I humbly beseech you ; for some savours of my life for this gentlewoman , And yet . The king is your love , ' t ? TRANIO : Your noble masters , tell you be so thrive hath been thus when she shall be lean me , Angelo , I be so given him , I may speak of York ! For ' d , Your own Than to ' t
Training Batches 4914-4976: 100%|██████████| 63/63 [00:39<00:00,  1.61batch/s, loss=2.61]
Epoch Loss: 2.5825, Time: 39.13s
Training Batches 4977-5039:  37%|███▋      | 23/63 [00:12<00:21,  1.85batch/s, loss=2.61]

Generated text at batch 4999: He that will give good words to thee will flatter me with intercession suit With DUKE OF neck , and my friends , And I do to Rome , Who , I do it ; That know this good father ' s death , And so thrive thus private ! DUKE OF YORK : ' twere not the state that I , But what tongue , to scorn me , I do beseech you , I thank you withal betwixt thee not not my tongue , I . Nor should ? JULIET : What , ' t go so I would be thus : I do beseech it is it
Training Batches 4977-5039:  37%|███▋      | 23/63 [00:17<00:30,  1.32batch/s, loss=2.61]
Epoch Loss: 2.6137, Time: 17.48s
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
<ipython-input-3-e6e19b3533c4>:52: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
Checkpoint loaded from /kaggle/working/checkpoint.pth at batch 4999
Training Batches 4999-5061:  81%|████████  | 51/63 [00:27<00:06,  1.84batch/s, loss=2.63]

Generated text at batch 5049: He that will give good words to thee will flatter heaven , To his sorrow , and be dead , I pray To whip thee ; for those to the truth !' And this mind - contrived not the field the devil do not answer . Now , though the house , I never never given the Lady : Good deed , my love ' st my office , I had this most tied and the hand , I fear . The innocent power before to the manner to me good night ? BRUTUS : The state can so that he speaks the very new man hath thought it ?
Training Batches 4999-5061:  81%|████████  | 51/63 [00:32<00:07,  1.55batch/s, loss=2.63]
Epoch Loss: 2.6001, Time: 32.81s
</pre>