# Pronunciation Transformer
This project builds a transformer model and all its components using TensorFlow to predict the IPA pronunciation of words. This is a signiciant extension and reorganisation of the [TensorFlow transformer tutorial notebook](https://www.tensorflow.org/text/tutorials/transformer) into a reusable Python package, developing a new transformer pronunciation model for British English.

# Background

## International Phonetic Alphabet (IPA) 
The IPA is a system for phonetically representing words. For example, in British pronunciation:
* House - /ha‍ʊs/
* Computer - /kəmˈpjuːtə/
* Ostrich - /ˈɒstrɪtʃ/

Here each IPA symbol has a unique pronunciation, which can be combined to pronounce words.

For more information: [International Phonetic Alphabet (Wikipedia)](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet)

## Transformers
A [transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) is a deep learning model, which uses the notion of self-attention in a highly parallelisable architecture. Transformers have largely been applied to sequence modelling, however the application domain is expanding.

# Getting Started

## Envionment
Python 3.9.13 on a Macbook Pro with an M1 Pro.

## Data
A British English IPA pronunciation dictionary, courtesy of [open-dict-data](https://github.com/open-dict-data/ipa-dict).

```
aah	/ˈɑː/
aardvark	/ˈɑːdvɑːk/
aardvarks	/ˈɑːdvɑːks/
aardwolf	/ˈɑːdwʊlf/
aba	/ɐbˈæ/
...
zydeco	/za‍ɪdˈiːkə‍ʊ/
zygoma	/za‍ɪɡˈə‍ʊmɐ/
zygote	/zˈa‍ɪɡə‍ʊt/
zygotic	/za‍ɪɡˈɒtɪk/
z	/zˈɛd/
```

## Installation
1. Clone the repository
```
git clone https://github.com/alfiebeard/pronunciation-transformer.git
```
2. Create virtual environment
```
virtualenv pronunciation_transformer_venv
source pronunciation_transformer_venv/bin/activate
```
Or with conda
```
conda create --name pronunciation_transformer python=3.10.5
source activate pronunciation_transformer
```
3. Install requirements.txt
```
pip install -r requirements.txt
```

## Running Instructions
There are several example scripts which can be used to train a pronunciation transformer, evaluate a trained model and run the model on a set of words.

### Training a model
```
python examples/training.py
```

### Evaluating a model
```
python examples/testing.py
```

### Running a model
```
python examples/run.py hello world
```
This will give predictions for the words hello and world. For example, hello -> heləʊ, world -> wɜːld.

## Key Steps
* [Preprocessing the IPA Dictionary](data/preprocessing/preprocess_ipa_dict.py) - creates the IPA datasets from the .txt dictionary and formats it for our needs.
* [Embeddings](data/embeddings/embedding.py) - creates the embeddings and stores the saved embeddings.
* [Model](model) - the transformer model.
* [Model Training](model_training) - train the model.
* [Model Evaluation](model_evaluation) - test a model and use it to make predictions.

# Training the Transformer

## Hyperparamters
```
Epochs = 5
Batch Size = 1000
Number of Layers = 2
Dimension of Model = 256
Number of Tranformer Heads = 8
Dimension of Hidden Layer in Feedforward Network = 1024 
Positional Encoding of Input = 500
Positional Encoding of Target = 2000
Positional Encoding Denominator = 1000
Dropout Rate = 0.05
Beta 1 = 0.9
Beta 2 = 0.98
Epsilon = 1e-9
```

## Metrics Used
* Accuracy - the accuracy of the model in predicting the next item of the sequence from the previous items.
* Complete Match Accuracy - the accuracy of the model in predicting the entire sequence correctly.

## Results
### Summary
Over a period of 20 epochs (split into 4 separate training cycles) we are able to achieve an accuracy of 97.5% on the validation set, with a complete match accuracy of 83.4%.

```
================ Validation Summary ================
Validation Set Accuracy: 0.9754527807235718
Validation Set Complete Match Accuracy 0.8341411352157593
```

The 4 checkpoints (epochs 5, 10, 15 and 20) are included in [saved_model_outputs](saved_model_outputs). The complete model is also stored there, but under the [saved_models](saved_model_outputs/saved_models) directory.

### Full Training Output

Training for 5 epochs (0-5).

```
================ Training ================
Epoch 1 Batch 0 Loss 4.6410231590271 Accuracy 0.009765625
Epoch 1 Batch 100 Loss 3.7030532360076904 Accuracy 0.09677629172801971
Epoch 1 Batch 200 Loss 3.1155014038085938 Accuracy 0.2143237292766571
Epoch 1 Batch 300 Loss 2.53769850730896 Accuracy 0.34610268473625183
Epoch 1 Batch 400 Loss 2.1368627548217773 Accuracy 0.4384024143218994
Epoch 1 Batch 500 Loss 1.8612319231033325 Accuracy 0.503156304359436
Epoch 1 Batch 600 Loss 1.6575535535812378 Accuracy 0.5519557595252991
Epoch 1 Batch 700 Loss 1.5007566213607788 Accuracy 0.5899561047554016
Epoch 1 Batch 800 Loss 1.3758538961410522 Accuracy 0.6207452416419983
Epoch 1 Loss 1.3614025115966797 Accuracy 0.6242577433586121
Time taken for epoch 1: 95.15 secs

Epoch 2 Batch 0 Loss 0.4389381408691406 Accuracy 0.8478701710700989
Epoch 2 Batch 100 Loss 0.45551246404647827 Accuracy 0.8518213033676147
Epoch 2 Batch 200 Loss 0.4374716281890869 Accuracy 0.8568298816680908
Epoch 2 Batch 300 Loss 0.41863158345222473 Accuracy 0.8622678518295288
Epoch 2 Batch 400 Loss 0.4036501348018646 Accuracy 0.8663510680198669
Epoch 2 Batch 500 Loss 0.3926445543766022 Accuracy 0.8697377443313599
Epoch 2 Batch 600 Loss 0.38310831785202026 Accuracy 0.8726829886436462
Epoch 2 Batch 700 Loss 0.37339088320732117 Accuracy 0.8759543895721436
Epoch 2 Batch 800 Loss 0.3650655746459961 Accuracy 0.8786622285842896
Epoch 2 Loss 0.3642721176147461 Accuracy 0.8789783120155334
Time taken for epoch 2: 74.92 secs

Epoch 3 Batch 0 Loss 0.30273550748825073 Accuracy 0.8903846144676208
Epoch 3 Batch 100 Loss 0.3036516606807709 Accuracy 0.8994345664978027
Epoch 3 Batch 200 Loss 0.29887792468070984 Accuracy 0.9010671377182007
Epoch 3 Batch 300 Loss 0.2929508686065674 Accuracy 0.9025700092315674
Epoch 3 Batch 400 Loss 0.2872670292854309 Accuracy 0.9037251472473145
Epoch 3 Batch 500 Loss 0.28406545519828796 Accuracy 0.904632031917572
Epoch 3 Batch 600 Loss 0.2818387448787689 Accuracy 0.9056142568588257
Epoch 3 Batch 700 Loss 0.2787051200866699 Accuracy 0.9065877795219421
Epoch 3 Batch 800 Loss 0.2760425806045532 Accuracy 0.907431423664093
Epoch 3 Loss 0.27597129344940186 Accuracy 0.9074759483337402
Time taken for epoch 3: 70.85 secs

Epoch 4 Batch 0 Loss 0.2208738476037979 Accuracy 0.9224490523338318
Epoch 4 Batch 100 Loss 0.26795998215675354 Accuracy 0.9104363322257996
Epoch 4 Batch 200 Loss 0.2590056359767914 Accuracy 0.9136025905609131
Epoch 4 Batch 300 Loss 0.2559898793697357 Accuracy 0.9146977663040161
Epoch 4 Batch 400 Loss 0.254116028547287 Accuracy 0.9153851270675659
Epoch 4 Batch 500 Loss 0.25363975763320923 Accuracy 0.9156538844108582
Epoch 4 Batch 600 Loss 0.25380462408065796 Accuracy 0.9157600998878479
Epoch 4 Batch 700 Loss 0.25420019030570984 Accuracy 0.9158895611763
Epoch 4 Batch 800 Loss 0.2540144622325897 Accuracy 0.9161394238471985
Epoch 4 Loss 0.2542406916618347 Accuracy 0.9160473346710205
Time taken for epoch 4: 73.64 secs

Epoch 5 Batch 0 Loss 0.34441837668418884 Accuracy 0.8878143429756165
Epoch 5 Batch 100 Loss 0.2510583996772766 Accuracy 0.915976881980896
Epoch 5 Batch 200 Loss 0.2490094006061554 Accuracy 0.9172026515007019
Epoch 5 Batch 300 Loss 0.25079289078712463 Accuracy 0.9168270826339722
Epoch 5 Batch 400 Loss 0.2502271234989166 Accuracy 0.9170310497283936
Epoch 5 Batch 500 Loss 0.24851684272289276 Accuracy 0.9176143407821655
Epoch 5 Batch 600 Loss 0.25237372517585754 Accuracy 0.9167242646217346
Epoch 5 Batch 700 Loss 0.2530724108219147 Accuracy 0.9163415431976318
Epoch 5 Batch 800 Loss 0.2545514702796936 Accuracy 0.9161346554756165
Epoch 5 Loss 0.2547779679298401 Accuracy 0.9160573482513428
Time taken for epoch 5: 81.00 secs

================ Training Summary ================
Total training time: 395.55 secs
Training accuracy: 0.9160573482513428

================ Evaluating Validation Set ================
Batch 0 Accuracy 0.9200863838195801
Batch 0 Complete Match Accuracy 0.640625
Batch 100 Accuracy 0.9198541045188904
Batch 100 Complete Match Accuracy 0.5491955280303955

================ Validation Summary ================
Validation Set Accuracy: 0.9199203252792358
Validation Set Complete Match Accuracy 0.5491589307785034
Time taken: 17.08 secs
```

Training for another 5 epochs (5-10).

```
================ Training ================
Epoch 1 Batch 0 Loss 0.36452531814575195 Accuracy 0.88416987657547
Epoch 1 Batch 100 Loss 0.25767555832862854 Accuracy 0.9147621989250183
Epoch 1 Batch 200 Loss 0.2585058808326721 Accuracy 0.9157573580741882
Epoch 1 Batch 300 Loss 0.2502050995826721 Accuracy 0.9180423617362976
Epoch 1 Batch 400 Loss 0.24653413891792297 Accuracy 0.9189462065696716
Epoch 1 Batch 500 Loss 0.2437353879213333 Accuracy 0.9198021292686462
Epoch 1 Batch 600 Loss 0.24304936826229095 Accuracy 0.9199053049087524
Epoch 1 Batch 700 Loss 0.2409413605928421 Accuracy 0.9204144477844238
Epoch 1 Batch 800 Loss 0.2399691939353943 Accuracy 0.9207389950752258
Epoch 1 Loss 0.24013777077198029 Accuracy 0.9207513928413391
Time taken for epoch 1: 94.80 secs

Epoch 2 Batch 0 Loss 0.2530861496925354 Accuracy 0.9158512353897095
Epoch 2 Batch 100 Loss 0.23115617036819458 Accuracy 0.9232528805732727
Epoch 2 Batch 200 Loss 0.22378692030906677 Accuracy 0.9258004426956177
Epoch 2 Batch 300 Loss 0.22026725113391876 Accuracy 0.9269298315048218
Epoch 2 Batch 400 Loss 0.21766000986099243 Accuracy 0.9275316596031189
Epoch 2 Batch 500 Loss 0.21505852043628693 Accuracy 0.9284000396728516
Epoch 2 Batch 600 Loss 0.21501131355762482 Accuracy 0.9284599423408508
Epoch 2 Batch 700 Loss 0.2138768434524536 Accuracy 0.9288123846054077
Epoch 2 Batch 800 Loss 0.21156074106693268 Accuracy 0.9294626116752625
Epoch 2 Loss 0.21154330670833588 Accuracy 0.9294481873512268
Time taken for epoch 2: 73.16 secs

Epoch 3 Batch 0 Loss 0.2640385031700134 Accuracy 0.9229323267936707
Epoch 3 Batch 100 Loss 0.21100321412086487 Accuracy 0.9307000041007996
Epoch 3 Batch 200 Loss 0.20395268499851227 Accuracy 0.9326699376106262
Epoch 3 Batch 300 Loss 0.20183268189430237 Accuracy 0.9327517151832581
Epoch 3 Batch 400 Loss 0.19778259098529816 Accuracy 0.9336802363395691
Epoch 3 Batch 500 Loss 0.19447125494480133 Accuracy 0.9346235394477844
Epoch 3 Batch 600 Loss 0.19317856431007385 Accuracy 0.9352128505706787
Epoch 3 Batch 700 Loss 0.19207262992858887 Accuracy 0.9354966878890991
Epoch 3 Batch 800 Loss 0.1907723844051361 Accuracy 0.9359519481658936
Epoch 3 Loss 0.19092850387096405 Accuracy 0.9359351992607117
Time taken for epoch 3: 70.92 secs

Epoch 4 Batch 0 Loss 0.1596187800168991 Accuracy 0.9370229244232178
Epoch 4 Batch 100 Loss 0.1834942102432251 Accuracy 0.9391587972640991
Epoch 4 Batch 200 Loss 0.1798464059829712 Accuracy 0.9398580193519592
Epoch 4 Batch 300 Loss 0.17515364289283752 Accuracy 0.9411622881889343
Epoch 4 Batch 400 Loss 0.17562341690063477 Accuracy 0.9407358765602112
Epoch 4 Batch 500 Loss 0.17446577548980713 Accuracy 0.9411315321922302
Epoch 4 Batch 600 Loss 0.17269518971443176 Accuracy 0.9417367577552795
Epoch 4 Batch 700 Loss 0.173312708735466 Accuracy 0.9415814876556396
Epoch 4 Batch 800 Loss 0.1732020527124405 Accuracy 0.9416293501853943
Epoch 4 Loss 0.17310850322246552 Accuracy 0.9416507482528687
Time taken for epoch 4: 71.35 secs

Epoch 5 Batch 0 Loss 0.16481472551822662 Accuracy 0.9516441226005554
Epoch 5 Batch 100 Loss 0.1745399534702301 Accuracy 0.9410832524299622
Epoch 5 Batch 200 Loss 0.1655188500881195 Accuracy 0.9437463879585266
Epoch 5 Batch 300 Loss 0.1628863662481308 Accuracy 0.9446398019790649
Epoch 5 Batch 400 Loss 0.16047905385494232 Accuracy 0.9450269341468811
Epoch 5 Batch 500 Loss 0.15961618721485138 Accuracy 0.9452275633811951
Epoch 5 Batch 600 Loss 0.15987750887870789 Accuracy 0.9451008439064026
Epoch 5 Batch 700 Loss 0.1590779572725296 Accuracy 0.9454220533370972
Epoch 5 Batch 800 Loss 0.1578681766986847 Accuracy 0.9457814693450928
Epoch 5 Loss 0.15799425542354584 Accuracy 0.9457114338874817
Time taken for epoch 5: 80.98 secs

================ Training Summary ================
Total training time: 391.20 secs
Training accuracy: 0.9457114338874817

================ Evaluating Validation Set ================
Batch 0 Accuracy 0.9543726444244385
Batch 0 Complete Match Accuracy 0.71875
Batch 100 Accuracy 0.9562014937400818
Batch 100 Complete Match Accuracy 0.7198328971862793

================ Validation Summary ================
Validation Set Accuracy: 0.9561555981636047
Validation Set Complete Match Accuracy 0.7201287150382996
Time taken: 17.67 secs
```

Training for another 5 epochs (10-15).

```
================ Training ================
Epoch 1 Batch 0 Loss 0.13874036073684692 Accuracy 0.9544554948806763
Epoch 1 Batch 100 Loss 0.16413012146949768 Accuracy 0.9445835947990417
Epoch 1 Batch 200 Loss 0.15466439723968506 Accuracy 0.9463229179382324
Epoch 1 Batch 300 Loss 0.15097644925117493 Accuracy 0.9477244019508362
Epoch 1 Batch 400 Loss 0.14681193232536316 Accuracy 0.948978066444397
Epoch 1 Batch 500 Loss 0.1471846103668213 Accuracy 0.9490771889686584
Epoch 1 Batch 600 Loss 0.1454925686120987 Accuracy 0.9495806694030762
Epoch 1 Batch 700 Loss 0.14846211671829224 Accuracy 0.9487585425376892
Epoch 1 Batch 800 Loss 0.14674507081508636 Accuracy 0.9492897987365723
Epoch 1 Loss 0.14673426747322083 Accuracy 0.9493002891540527
Time taken for epoch 1: 94.82 secs

Epoch 2 Batch 0 Loss 0.11035270243883133 Accuracy 0.9612545967102051
Epoch 2 Batch 100 Loss 0.14267592132091522 Accuracy 0.95111483335495
Epoch 2 Batch 200 Loss 0.13913178443908691 Accuracy 0.952476978302002
Epoch 2 Batch 300 Loss 0.1366795003414154 Accuracy 0.9528679847717285
Epoch 2 Batch 400 Loss 0.13610656559467316 Accuracy 0.9528120756149292
Epoch 2 Batch 500 Loss 0.13548479974269867 Accuracy 0.9529879093170166
Epoch 2 Batch 600 Loss 0.13612264394760132 Accuracy 0.9528335928916931
Epoch 2 Batch 700 Loss 0.13602063059806824 Accuracy 0.9528961181640625
Epoch 2 Batch 800 Loss 0.13483566045761108 Accuracy 0.9533259272575378
Epoch 2 Loss 0.13535946607589722 Accuracy 0.9532360434532166
Time taken for epoch 2: 73.04 secs

Epoch 3 Batch 0 Loss 0.1656751036643982 Accuracy 0.9451220035552979
Epoch 3 Batch 100 Loss 0.1440613567829132 Accuracy 0.950476884841919
Epoch 3 Batch 200 Loss 0.13691598176956177 Accuracy 0.9527650475502014
Epoch 3 Batch 300 Loss 0.13584306836128235 Accuracy 0.9529076218605042
Epoch 3 Batch 400 Loss 0.13112062215805054 Accuracy 0.9541266560554504
Epoch 3 Batch 500 Loss 0.12968257069587708 Accuracy 0.9548641443252563
Epoch 3 Batch 600 Loss 0.13048188388347626 Accuracy 0.9550032615661621
Epoch 3 Batch 700 Loss 0.12941475212574005 Accuracy 0.9553627967834473
Epoch 3 Batch 800 Loss 0.12823443114757538 Accuracy 0.955652117729187
Epoch 3 Loss 0.1282515972852707 Accuracy 0.9555855989456177
Time taken for epoch 3: 70.42 secs

Epoch 4 Batch 0 Loss 0.13711120188236237 Accuracy 0.9545454978942871
Epoch 4 Batch 100 Loss 0.14133544266223907 Accuracy 0.9518983960151672
Epoch 4 Batch 200 Loss 0.12580250203609467 Accuracy 0.9564070701599121
Epoch 4 Batch 300 Loss 0.12267059832811356 Accuracy 0.9576443433761597
Epoch 4 Batch 400 Loss 0.11972681432962418 Accuracy 0.9581366777420044
Epoch 4 Batch 500 Loss 0.11951066553592682 Accuracy 0.9582231044769287
Epoch 4 Batch 600 Loss 0.1194903776049614 Accuracy 0.9582861661911011
Epoch 4 Batch 700 Loss 0.11861845850944519 Accuracy 0.9585494995117188
Epoch 4 Batch 800 Loss 0.118403360247612 Accuracy 0.958512008190155
Epoch 4 Loss 0.11840526759624481 Accuracy 0.9584779143333435
Time taken for epoch 4: 79.41 secs

Epoch 5 Batch 0 Loss 0.10157748311758041 Accuracy 0.9657657742500305
Epoch 5 Batch 100 Loss 0.12399381399154663 Accuracy 0.9561144113540649
Epoch 5 Batch 200 Loss 0.11569672077894211 Accuracy 0.9588165879249573
Epoch 5 Batch 300 Loss 0.11417771130800247 Accuracy 0.959717869758606
Epoch 5 Batch 400 Loss 0.11168773472309113 Accuracy 0.9601669311523438
Epoch 5 Batch 500 Loss 0.11285389214754105 Accuracy 0.9600953459739685
Epoch 5 Batch 600 Loss 0.11296840012073517 Accuracy 0.9601888060569763
Epoch 5 Batch 700 Loss 0.11253008246421814 Accuracy 0.9604296684265137
Epoch 5 Batch 800 Loss 0.11204471439123154 Accuracy 0.9604631662368774
Epoch 5 Loss 0.11217739433050156 Accuracy 0.9604528546333313
Time taken for epoch 5: 77.80 secs

================ Training Summary ================
Total training time: 395.50 secs
Training accuracy: 0.9604528546333313

================ Evaluating Validation Set ================
Batch 0 Accuracy 0.975708544254303
Batch 0 Complete Match Accuracy 0.828125
Batch 100 Accuracy 0.9672425985336304
Batch 100 Complete Match Accuracy 0.7810952663421631

================ Validation Summary ================
Validation Set Accuracy: 0.9673290252685547
Validation Set Complete Match Accuracy 0.782127320766449
Time taken: 16.64 secs
```

Training for another 5 epochs (15-20).

```
================ Training ================
Epoch 1 Batch 0 Loss 0.09314244985580444 Accuracy 0.9654510021209717
Epoch 1 Batch 100 Loss 0.11436515301465988 Accuracy 0.9591019749641418
Epoch 1 Batch 200 Loss 0.10973840951919556 Accuracy 0.9603314399719238
Epoch 1 Batch 300 Loss 0.10736966133117676 Accuracy 0.9616954326629639
Epoch 1 Batch 400 Loss 0.10691281408071518 Accuracy 0.9619933366775513
Epoch 1 Batch 500 Loss 0.10857434570789337 Accuracy 0.9617375135421753
Epoch 1 Batch 600 Loss 0.10726068913936615 Accuracy 0.962134838104248
Epoch 1 Batch 700 Loss 0.10658734291791916 Accuracy 0.9623990654945374
Epoch 1 Batch 800 Loss 0.10568835586309433 Accuracy 0.9626339077949524
Epoch 1 Loss 0.10571441799402237 Accuracy 0.9626518487930298
Time taken for epoch 1: 96.26 secs

Epoch 2 Batch 0 Loss 0.10339641571044922 Accuracy 0.9635627865791321
Epoch 2 Batch 100 Loss 0.11741337925195694 Accuracy 0.9603142738342285
Epoch 2 Batch 200 Loss 0.1082768589258194 Accuracy 0.9629889130592346
Epoch 2 Batch 300 Loss 0.10433379560709 Accuracy 0.9634408354759216
Epoch 2 Batch 400 Loss 0.10219955444335938 Accuracy 0.9637761116027832
Epoch 2 Batch 500 Loss 0.10350213944911957 Accuracy 0.9636639356613159
Epoch 2 Batch 600 Loss 0.10314030200242996 Accuracy 0.9638383984565735
Epoch 2 Batch 700 Loss 0.10200832784175873 Accuracy 0.9640987515449524
Epoch 2 Batch 800 Loss 0.1013273298740387 Accuracy 0.964302659034729
Epoch 2 Loss 0.10138605535030365 Accuracy 0.9642828106880188
Time taken for epoch 2: 78.14 secs

Epoch 3 Batch 0 Loss 0.12408816814422607 Accuracy 0.9691715240478516
Epoch 3 Batch 100 Loss 0.1109481230378151 Accuracy 0.9609101414680481
Epoch 3 Batch 200 Loss 0.10584922879934311 Accuracy 0.9629619121551514
Epoch 3 Batch 300 Loss 0.09955091029405594 Accuracy 0.9649907350540161
Epoch 3 Batch 400 Loss 0.09616691619157791 Accuracy 0.9657570123672485
Epoch 3 Batch 500 Loss 0.09538984298706055 Accuracy 0.9658546447753906
Epoch 3 Batch 600 Loss 0.09554430842399597 Accuracy 0.9658398628234863
Epoch 3 Batch 700 Loss 0.09689260274171829 Accuracy 0.9657236933708191
Epoch 3 Batch 800 Loss 0.09612607210874557 Accuracy 0.9658903479576111
Epoch 3 Loss 0.0959395244717598 Accuracy 0.9659553170204163
Time taken for epoch 3: 73.55 secs

Epoch 4 Batch 0 Loss 0.08746668696403503 Accuracy 0.9600798487663269
Epoch 4 Batch 100 Loss 0.09726454317569733 Accuracy 0.9662242531776428
Epoch 4 Batch 200 Loss 0.0940609723329544 Accuracy 0.9666376709938049
Epoch 4 Batch 300 Loss 0.09371250867843628 Accuracy 0.9668905138969421
Epoch 4 Batch 400 Loss 0.09370464831590652 Accuracy 0.9667779803276062
Epoch 4 Batch 500 Loss 0.09227453172206879 Accuracy 0.9673640131950378
Epoch 4 Batch 600 Loss 0.09149723500013351 Accuracy 0.9676249623298645
Epoch 4 Batch 700 Loss 0.09225673973560333 Accuracy 0.9673671126365662
Epoch 4 Batch 800 Loss 0.0915205106139183 Accuracy 0.9675822854042053
Epoch 4 Loss 0.0914958119392395 Accuracy 0.9676037430763245
Time taken for epoch 4: 72.23 secs

Epoch 5 Batch 0 Loss 0.15599055588245392 Accuracy 0.9571694731712341
Epoch 5 Batch 100 Loss 0.09708744287490845 Accuracy 0.9653623104095459
Epoch 5 Batch 200 Loss 0.09047956764698029 Accuracy 0.9678711891174316
Epoch 5 Batch 300 Loss 0.0886916071176529 Accuracy 0.9683834910392761
Epoch 5 Batch 400 Loss 0.08630789816379547 Accuracy 0.9688218832015991
Epoch 5 Batch 500 Loss 0.08724942058324814 Accuracy 0.9687595963478088
Epoch 5 Batch 600 Loss 0.08698450773954391 Accuracy 0.9689671993255615
Epoch 5 Batch 700 Loss 0.08664815127849579 Accuracy 0.9690430760383606
Epoch 5 Batch 800 Loss 0.08638797700405121 Accuracy 0.9690320491790771
Epoch 5 Loss 0.08658567070960999 Accuracy 0.9689973592758179
Time taken for epoch 5: 73.58 secs

================ Training Summary ================
Total training time: 393.77 secs
Training accuracy: 0.9689973592758179

================ Evaluating Validation Set ================
Batch 0 Accuracy 0.9769874215126038
Batch 0 Complete Match Accuracy 0.828125
Batch 100 Accuracy 0.9754065871238708
Batch 100 Complete Match Accuracy 0.8338490128517151

================ Validation Summary ================
Validation Set Accuracy: 0.9754527807235718
Validation Set Complete Match Accuracy 0.8341411352157593
Time taken: 18.65 secs
```

# Extensions

## Speech Model Extension
This [extension](speech_model) is an experiment to see whether the predictions can be used to generate speech. The speech is interpretable in most cases, but there is more work to be done on the synthesis of it.

"I have a dream" fed into the transformer and then converted into IPA sounds, [listen here](speech_model/saved_audio_predictions/i_have_a_dream.mp3).

IPA data courtesy of [Glitzyken/ipa-sounds](https://github.com/Glitzyken/ipa-sounds).

## Others
* Introduce suprasegmentals, such as primary and secondary stresses.
* Expansion into different languages and dialects.
* Further tuning of the transformer.


# License
Licensed under the [MIT License](LICENSE.md).
