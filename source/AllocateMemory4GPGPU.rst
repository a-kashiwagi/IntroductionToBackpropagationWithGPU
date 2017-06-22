=============================================
汎用GPUにおける結合荷重及び関連値の確保と保持
=============================================

:題名: 汎用GPUを利用するにあたっての、結合荷重及び関連値の確保と保持についての解説
:著者: 柏木 明博
:作成日: 2017年6月14日

不連続な領域としてではなく、連続した領域としてメモリを確保する
==============================================================

ここでは、汎用GPUを利用するにあたって、結合荷重 :math:`w` やバイアス :math:`b` 、
出力 :math:`z` や誤差 :math:`d` の確保と保持の仕方について、説明します。

CPU上では、適ほど構造体を定義し、その構造体の要素としてポインタを用意すること
で、インスタンス作成時にそれぞれ必要なだけ、malloc()することができます。この
状態でそのポインタの実体に対して計算を行えば利用可能ですが、メモリ上は連続し
た領域にメモリが確保されておらず、断片化しています。汎用GPUを利用する場合には、
使用するメモリ領域を汎用GPUとCPUとの間で転送しなければなりませんが、断片化し
た状態では転送が非常に煩雑となり、処理速度にも影響してきます。転送回数を減ら
すため、連続したメモリ領域を利用できるように工夫が必要です。

今回は、Neural Networkの層数と、各層のユニット数から、必要なメモリ量を事前に
計算し、それを一回限りの malloc()で確保することで、汎用GPUとの転送の効率化を
図っています。


一覧表1.データ構造

#. 構造体への保持用領域
#. 全体の層数 :math:`l` 保持用領域
#. 層ごとの 出力値 :math:`z` の数 保持領域
#. 層ごとの バイアス値 :math:`b` の数 保持領域
#. 層ごとの 結合加重 :math:`w` の数 保持領域
#. 層ごとの 誤差 :math:`d` の数 保持領域
#. 層ごとの バイアス誤差 :math:`db` の数 保持領域
#. 層ごとの実際の 出力値 :math:`z` 保持領域
#. 層ごとの実際の バイアス値 :math:`b` 保持領域
#. 層ごとの実際の 結合加重 :math:`w` 保持領域
#. 層ごとの実際の 誤差 :math:`d` 保持領域
#. 層ごとの実際の バイアス誤差 :math:`db` 保持領域

一覧表1に挙げた各要素に必要なメモリ領域を加算し、合計値でmalloc()する事で、
連続した領域を確保します。具体的なコードで示すと、以下のようにります。

LIST 1. NEUTON_T構造体

.. code-block:: cpp

					// Neuron structure
	typedef struct neuron_t{

		double **z;
					// value  of z
		long *z_num; 
					// number of z
		double **b;
					// value  of b
		long *b_num; 
					// number of b
		double **w;
					// value  of w
		long *w_num; 
					// number of w
		double **d;
					// value  of d
		long *d_num; 
					// number of d
		double **db;
					// value  of db
		long *db_num;
					// number of db
	} NEURON_T;

LIST 2. 変数宣言

.. code-block:: cpp

	cudaError_t err;
					// Error code of cuda
	long phase;
					// Number of phase
	long size;
					// Size of memory
	long cur;
					// Cursor
	long *mem_cpu_p;

	NEURON_T *n;

LIST 3. サイズ計算

.. code-block:: cpp

					// Long type pointer of cpu side memory
	size = 0;
					// Init value of size 

					// Add number of size of NEURON_T
	size += sizeof(NEURON_T);
					// Add number of size of l_num
	size += sizeof(long);
					// Add number of 1).z_num, 2).b_num,
					// 3).w_num, 4).d_num, 5).db_num
	size += sizeof(long) * 5 * l_num;

	for(phase = 0; phase < l_num; phase++){
					// Calculate an all size
		size += sizeof(double) * (
			   z_num[phase]
			+  b_num[phase]
			+  w_num[phase]
			+  d_num[phase]
			+ db_num[phase]
		);
	}

ここでは、一覧表1に挙げた順にメモリサイズを計算しています。まず、0でリセットし、
先頭部分にNEURON_T構造体のサイズ分確保します。そして、層数、各層における
:math:`z,b,w,d,db` の数を保存する為のサイズを確保し、 :math:`z,b,w,d,db` の実際
の値を保存するサイズを加算します。


LIST 4. メモリ確保

.. code-block:: cpp

	*mem_cpu = (void *)malloc( size );
					// Memory allocate at CPU
	if( mem_cpu == NULL ){
		return( -1 );
	}

	err = cudaMalloc( (void**)&(*mem_dev), size );
					// Memory allocate at GPU
	if( err != cudaSuccess){
		return( -2 );
	}

計算して得た必要なメモリサイズを用いて、CPU側と汎用GPU側それぞれに連続した領
域を確保します。連続した領域は、一覧表1の先頭「1.構造体への保持用領域」の要
素である各値へのポインタへ再割り当てすることで、使用可能となります。再割り当
て処理は、汎用GPU上においても、CPU上でも同様です。

構造体への再割当て
==================

各種計算用関数から利用し易いように、構造体への再割り当てを行います。再割り当
ては、上記「一覧表1.構造体要素」を再計算し、それぞれの保持領域への先頭アドレ
スを「1.構造体への保持用領域」の要素である各値へのポインタへ格納し直します。

LIST 5. 引数の取得

.. code-block:: cpp

	__device__ __host__ NEURON_T *set_instance(
		long    l_num,
		void **mem
	){

LIST 6. 変数宣言

.. code-block:: cpp

	NEURON_T *n;
					// Pointer of liner memory for long
	long *mem_long;
					// Pointer of liner memory for double
	double *mem_double;
					// Counter for cursor
	long phase_len;
					// Counter for phase
	long phase;

LIST 7. メモリ領域の確保

.. code-block:: cpp

					// Init a length at each phase
	phase_len = 0;
					// Set address of top
	n = (NEURON_T *)*mem;
					// Increment cursor
	phase_len++;
					// allocate memory for z,b,w,d,db
	n->z_num  = (long *)malloc( sizeof(long) * l_num );
	n->b_num  = (long *)malloc( sizeof(long) * l_num );
	n->w_num  = (long *)malloc( sizeof(long) * l_num );
	n->d_num  = (long *)malloc( sizeof(long) * l_num );
	n->db_num = (long *)malloc( sizeof(long) * l_num );

	n->z  = (double**)malloc( sizeof(double*) * l_num);
	n->b  = (double**)malloc( sizeof(double*) * l_num);
	n->w  = (double**)malloc( sizeof(double*) * l_num);
	n->d  = (double**)malloc( sizeof(double*) * l_num);
	n->db = (double**)malloc( sizeof(double*) * l_num);

LIST 8. メモリアドレスのポインタへの再割当て

.. code-block:: cpp

					// Set pointer address
					//                 for long array again
	mem_long = (long *)&n[phase_len];
					//mem_long = (long *)*mem;

					// Initialize a cursor
	phase_len = 0;
					// Get number of phases of this network
	l_num = mem_long[phase_len];
					// Increment pointer
	phase_len++;

	for(phase = 0; phase < l_num; phase++){

					// Get number of z at each phase
		n->z_num[phase] = mem_long[phase_len];
		phase_len++;
	}

	for(phase = 0; phase < l_num; phase++){

					// Get number of b at each phase
		n->b_num[phase] = mem_long[(phase_len)];
		phase_len++;
	}

	for(phase = 0; phase < l_num; phase++){

					// Get number of w at each phase
		n->w_num[phase] = mem_long[(phase_len)];
		phase_len++;
	}

	for(phase = 0; phase < l_num; phase++){

					// Get number of d at each phase
		n->d_num[phase] = mem_long[(phase_len)];
		phase_len++;
	}

	for(phase = 0; phase < l_num; phase++){

					// Get number of db at each phase
		n->db_num[phase] = mem_long[(phase_len)];
		phase_len++;
	}
					// Set pointer address for long array
	mem_double = (double *)&mem_long[phase_len];

                                        // Initialize a cursor
	phase_len = 0;

	for( phase = 0; phase < l_num; phase++ ){

					// Set pointer to an each variables

		n->z[phase] = &mem_double[(phase_len) + 0];
					// for z

		n->b[phase] = &mem_double[(phase_len) + n->z_num[phase]];
					// for b

		n->w[phase] = &mem_double[
			(phase_len) + n->z_num[phase] + n->b_num[phase]
		];
					// for w

		n->d[phase] = &mem_double[
			(phase_len)
			+ n->z_num[phase]
			+ n->b_num[phase]
			+ n->w_num[phase]
		];                      // for delta

		n->db[phase] = &mem_double[
			(phase_len)
			+ n->z_num[phase]
			+ n->b_num[phase]
			+ n->w_num[phase]
			+ n->d_num[phase]
		];                      // for delta of bias

		phase_len
			+= n->z_num[phase]
			+  n->b_num[phase]
			+  n->w_num[phase]
			+  n->d_num[phase]
			+  n->db_num[phase];
					// Calculate a size of each phase
	}

LIST 8の先頭部分で、NEURON_T構造体のアドレスを(long *)でキャストしてmem_long
ポインタへ代入していますが、お分かりの通り、phase_lenの値は加算されて1となっ
ているため、一覧表1の「1.構造体への保持用領域」の次の要素である「2.全体の層数
:math:`l` 保持領域」を指しています。この層数 :math:`l` を格納しているl_numは
long型の為、わざわざポインタをキャストしてlong型として取り出せるようにしてい
ます。C言語のポインタマジックです。そして、phase_lenを0でリセットした後、改め
てmem_long[]の先頭アドレスの値をl_numへ代入し、層数 :math:`l` を取り出します。
この関数set_instance()の引数void **memには、l_numの位置に層数がセットされて引
き渡されてきます。以降、 :math:`z,b,w,d,db` の各層の数を順に取り出してNEURON_T
構造体へ代入して行きます。途中、double型のmem_doubleポインタへキャストしてい
るところがありますが、これも先ほど説明した通り、double型の値を取り出すために、
わざわざ(double *)でキャストしています。以降、 :math:`z,b,w,d,db` の値を取り出
してNEURON_T構造体へ代入しています。

以上、メモリ転送を考慮した、汎用GPUとCPUにおける連続したメモリ領域の確保と保
持についての解説です。

