import scala.util.Random;
import breeze.plot._;

object OldStyle {

	def main(args: Array[String]) {

		//1層目のユニット数　入力次元数　//バイアスユニットを含む
		val input_dimension = 2;
		//隠れ層のユニット数　//バイアスユニットを含む
		val hidden_number = 8;
		//出力の次元数
		val output_signal_dimension = 1;
		//学習係数
		val eta = 0.1;
		val rand = new Random;
	
		//入力と隠れ層の間の重みをランダムに初期化
		//[入力ユニットN番目][隠れそうユニットM番目]
		var w1 = Array.ofDim[Double]( input_dimension ,(input_dimension max hidden_number)  ).map{ v => v.map{v2 => rand.nextDouble }};
		//隠れ層と出力層の間の重みをランダムに初期化
		var w2 = Array.ofDim[Double]( hidden_number , (hidden_number max output_signal_dimension)  ).map{ v => v.map{v2 => rand.nextDouble  }};
		
		val N = 100;
		val input_signals = (-1.0 to 1.0 by 2.0 / N);
		val inst_signals = input_signals.map{v => 0.5 *  (Math.sin( Math.PI * v) + 1)}; // サインカーブ
		//val inst_signals = input_signals.map{v => v * v}; //二次関数
		
		println(input_signals);
		println(inst_signals);
    
    for ( loop <- 1 to 10000)
    {
    	for ( signal_index <- 0 until input_signals.length)
    	{
    		//入力信号
    		var x1 = new Array[Double](input_dimension );
    		//教師信号
    		var inst = new Array[Double](output_signal_dimension);

    		//入力信号
    		x1(0) = input_signals(signal_index);
    		//バイアスユニットの値の設定 最後の要素に1を代入
    		x1(x1.length-1) = 1.0;
    		//正解信号
    		inst(0) = inst_signals(signal_index);

    		var error = Double.MaxValue;
    		//隠れ層の信号　順伝播で与えられる
    		var x2 = new Array[Double](hidden_number);
    		for( i <- 0 until x1.length){
    			for( j <- 0 until x2.length-1){
    				x2(j) += w1(i)(j) * x1(i);
    			}
    		}
    		//シグモイド関数
    		x2 = x2.map{v => 1 / ( 1 + Math.exp(-v))};
    		//バイアスユニット
    		x2(x2.length-1) = 1.0;

    		//順伝播で出力層を更新  scalaっぽく書き換えたい
    		//出力信号
    		var output_signal = new Array[Double](output_signal_dimension);
    		for ( i <- 0 until x2.length){
    			for ( j <- 0 until output_signal.length){
    				output_signal(j) += w2(i)(j) * x2(i);
    			}
    		}
    		//シグモイド関数
    		output_signal = output_signal.map{ v=> 1 / ( 1 + Math.exp(-v))}
    		error = (output_signal zip inst).map( t => 0.5 * (t._1 - t._2) * (t._1 - t._2)).sum;
    		//println(error);

    		//出力層 の誤差を計算
    		var epsi_out = new Array[Double](output_signal_dimension);
    		for(i <- 0 until epsi_out.length){
    			epsi_out(i) = (output_signal(i) - inst(i)) * output_signal(i) * ( 1 - output_signal(i) );
    		}
    		//隠れ層 の誤差を計算
    		var epsi_hidden = new Array[Double](hidden_number);
    		for ( i <- 0 until epsi_hidden.length){
    			var temp = 0.0;
    			for ( j <- 0 until epsi_out.length){
    				temp += w2(i)(j) * epsi_out(j);
    			}
    			epsi_hidden(i) = temp * x2(i) *( 1 - x2(i));
    		}

    		//重みの修正 入力→隠れ層
    		for(i <- 0 until x1.length){
    			for( j <- 0 until x2.length ){
    				w1(i)(j) -= eta * ( x1(i) * epsi_hidden(j) );
    			}
    		}

    		//重みの修正 隠れ層→出力層
    		for (i <- 0 until x2.length){
    			for ( j <- 0 until output_signal.length){
    				w2(i)(j) -= eta * ( x2(i) * epsi_out(j) ); 
    			}
    		}
    	}
    }
		
    /*
		println(w1.deep);
		println(w2.deep);
    */
		//学習結果の重みを使って出力
		def output(x:Double):Double = {
			//順伝播で出力を計算
			var x1 = new Array[Double](input_dimension);
			x1(0) = x;
			x1(x1.length-1) = 1.0;
      var x2 = new Array[Double](hidden_number);
      for( i <- 0 until x1.length){
    	  for( j <- 0 until x2.length-1){
    		  x2(j) += w1(i)(j) * x1(i);
    	  }
      }
      //シグモイド関数
      x2 = x2.map{v => 1 / ( 1 + Math.exp(-v))};
      //バイアスユニット
      x2(x2.length-1) = 1.0;

      //順伝播で出力層を更新  scalaっぽく書き換えたい
      //出力信号
      var output_signal = new Array[Double](output_signal_dimension);
      for ( i <- 0 until x2.length){
    	  for ( j <- 0 until output_signal.length){
    		  output_signal(j) += w2(i)(j) * x2(i);
    	  }
      }
  		output_signal = output_signal.map{ v=> 1 / ( 1 + Math.exp(-v))};
      output_signal(0);
		}
    val leaned_output_signal = input_signals.map{ v => output(v) };
    println(leaned_output_signal);
    
    val f = Figure();
    val p = f.subplot(0);
    p += plot(input_signals,inst_signals,'-');
    p += plot(input_signals,leaned_output_signal,'+');
    //f.saveas("square.png");
  }
}