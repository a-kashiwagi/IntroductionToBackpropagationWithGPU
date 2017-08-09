============
結果(Result)
============

:題名: 実行結果の確認
:著者: 柏木 明博
:作成日: 2017年8月4日

実行結果の確認
==============

冒頭で解説した「Forward Propagation」をプログラムにすると、以下のようになります。

LIST 1.引数と変数宣言

.. code-block:: c

	__global__ void calc_forward(
	        int loop_cnt,
	        long trg,
	        void *mem,
	        double *data,
	        long datb_num,
	        int debug
	){
	        int tid;
	                                        // thread id
	        long i_cnt;
	                                        // counter of input side
	        long j_cnt;
	                                        // counter of output side
	        NEURON_T *n;
	                                        // neuron structure
	        double *zi;
	                                        // pointer of  input side z
	        double *zj;
	                                        // pointer of output side z
	        long iphase;
	                                        // number of input phase
	        long jphase;
	                                        // number of output phase
	        double THETA;
	                                        // Number of θ
	        long uniti;
	        long unitj;

LIST 2.初期値の設定とGPUに関連した処理

.. code-block:: c

	                                        // set neuron instance
	        n = (NEURON_T *)mem;
	                                        // Set phase number for i and j
	        iphase = trg + 0;
	        jphase = trg + 1;

	        uniti = n->z_num[iphase];
	        unitj = n->z_num[jphase];

	        tid = blockIdx.x;
	                                        // Set block ID
	        if(tid > unitj - 1 || tid < 0){
	                                        // check for enable threads
	                return;
	        }
	                                        // Set number of θ
	        THETA = 0.0000001;
	                                        // set pointer for each value
	        zi = n->z[iphase];
	        zj = n->z[jphase];
	                                        // set pointer for each value
	        zi = n->z[iphase];
	        zj = n->z[jphase];

LIST 3.訓練データの設定

.. code-block:: c

	        if( trg == 0 ){
	                                        // set train data
	                for( i_cnt = 0; i_cnt < uniti; i_cnt++ ){

	                        zi[i_cnt] = data[i_cnt + (uniti * datb_num)];
	                }
	        }

LIST 4.Forward Propagation(順伝搬)の計算

.. code-block:: c

	                                        // set block id
	        j_cnt = blockIdx.x;

	        if(j_cnt < unitj){
	                                        // calculate forward
	                zj[j_cnt] = 0;

	                for( i_cnt = 0; i_cnt < uniti; i_cnt++ ){

	                        if( trg != 0 ){
	                                zj[j_cnt]
	                                        += n->w[jphase][i_cnt + (uniti * j_cnt)]
	                                        * sigmoid( zi[i_cnt] );
	                        }else{
	                                zj[j_cnt]
	                                        += n->w[jphase][i_cnt + (uniti * j_cnt)]
	                                        * ( zi[i_cnt] );
	                        }
	                }

	                if(trg == 1 && j_cnt == 0){

						// Debug write
				printf("%d,%ld,(%.12f,%.12f),%f\n",
					loop_cnt,
					datb_num,
					data[(n->z_num[0] * datb_num) + 0],
					data[(n->z_num[0] * datb_num) + 1],
					sigmoid(zj[j_cnt] + n->b[jphase][j_cnt] - THETA)
				);
	                }

	                zj[j_cnt] += n->b[jphase][j_cnt] - THETA;
	                //      = sigmoid( zj[j_cnt] + b[jphase][j_cnt] );
	        }
	                                        // Normal return
	        return;
	}

プログラムの構成は、前述の「Back Propagation」と同様のため、説明はいらないは
ずですが、動作確認用にデバッグライトが入っています。実行すると、下記のような
結果が得られます。

.. code-block:: c

	0,0,(0.000000000000,0.000000000000),0.467041
	9,1,(1.000000000000,0.000000000000),0.412412
	18,2,(0.000000000000,1.000000000000),0.483921
	27,3,(1.000000000000,1.000000000000),0.597794
	36,0,(0.000000000000,0.000000000000),0.507634
	45,1,(1.000000000000,0.000000000000),0.423022
	54,2,(0.000000000000,1.000000000000),0.478601
	63,3,(1.000000000000,1.000000000000),0.591965
	72,0,(0.000000000000,0.000000000000),0.512274
	81,1,(1.000000000000,0.000000000000),0.428474
	90,2,(0.000000000000,1.000000000000),0.473584
	99,3,(1.000000000000,1.000000000000),0.586651

	　　　　　　　　　　　・
	　　　　　　　　　　　・
	　　　　　　　　　　　・

	1602,2,(0.000000000000,1.000000000000),0.499124
	1611,3,(1.000000000000,1.000000000000),0.580877
	1620,0,(0.000000000000,0.000000000000),0.280471
	1629,1,(1.000000000000,0.000000000000),0.662751
	1638,2,(0.000000000000,1.000000000000),0.505142
	1647,3,(1.000000000000,1.000000000000),0.576334
	1656,0,(0.000000000000,0.000000000000),0.271702
	1665,1,(1.000000000000,0.000000000000),0.671107
	1674,2,(0.000000000000,1.000000000000),0.511839
	1683,3,(1.000000000000,1.000000000000),0.571101
	1692,0,(0.000000000000,0.000000000000),0.263545

	　　　　　　　　　　　・
	　　　　　　　　　　　・
	　　　　　　　　　　　・

	2907,3,(1.000000000000,1.000000000000),0.135156
	2916,0,(0.000000000000,0.000000000000),0.171739
	2925,1,(1.000000000000,0.000000000000),0.848447
	2934,2,(0.000000000000,1.000000000000),0.849482
	2943,3,(1.000000000000,1.000000000000),0.130012
	2952,0,(0.000000000000,0.000000000000),0.170890
	2961,1,(1.000000000000,0.000000000000),0.850712
	2970,2,(0.000000000000,1.000000000000),0.852260
	2979,3,(1.000000000000,1.000000000000),0.125324
	2988,0,(0.000000000000,0.000000000000),0.170024
	2997,1,(1.000000000000,0.000000000000),0.852847


一番左は、ループカウンタ、次がデータ番号、そして括弧で括られた部分が入力値、
一番右の列は、出力値です。括弧で括られた値を排他的論理和(XOR)の真理値表に合
わせて、一番右の値が同じ結果になっているか確認することが出きます。ループの
初期から段々と出力値が収束してくのが分かると思います。そして、最後の4行を見
ると、真理値表の通りになっている事を確認できます。


References
==========

#. 「深層学習(Deep Learning)」 岡谷貴之 講談社
#. 「実装ディープラーニング」 藤谷一弥 高原歩 オーム社
#. 「畳み込みニューラルネットワーク徹底解説 TensorFlowで学ぶディープラーニング入門」 中居悦司 マイナビ出版
#. 「CUDA BY EXAMPLE 汎用GPUプログラミング入門」 Jason Sanders Edward Kandrot インプレスジャパン
#. 「ニューロンの生物物理[第2版]」 宮川博義 井上雅司 丸善出版
#. 「脳・神経と行動」 佐藤真彦 岩波書店
#. 「ニューラルネットワーク(シリーズ非線形科学入門)」 吉冨康成 朝倉書店
#. 「C++とJavaでつくるニューラルネットワーク」 平野廣美 パーソナルメディア
#. 「人工知能学会 学会誌 Vol.29 No.1〜6」 人工知能学会 オーム社
#. 「Theano入門」 株式会社知能情報システム 吉岡琢 <http://www.chino-js.com/ja/tech/theano-rbm/>


