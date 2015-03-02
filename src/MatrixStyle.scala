import scala.util.Random;
import breeze.linalg._;
import breeze.plot._;
//import scalala.scalar._;

object MatrixStyle {

  def sigmoid(v:Double):Double = {
    1.0 / ( 1.0 + Math.exp(-v));
  }
  
  def main(args: Array[String]): Unit = {
    val signal_num = 2; //バイアス含めた入力次元数
    val hidden_num = 4; //隠れ層の次元数
    val output_num = 1;
    val eta = 0.1;
    
    val N = 100;
    val temp_input = ( -1.0 to 1.0 by 2.0 / N ).toArray;
    val temp_inst = temp_input.map { v => 0.5 * ( Math.sin ( Math.PI * v ) + 1)};
    val input = ((arr:Array[Double]) => {
      	var a = Array.ofDim[Double](arr.length, signal_num-1);
      	for( i <- 0 until arr.length){
      		a(i)(0) = arr(i);
      	}
      	a;
      }:Array[Array[Double]])(temp_input);
    
    val instraction = ((arr:Array[Double]) => {
      var a = Array.ofDim[Double](arr.length, output_num);
      for ( i <- 0 until arr.length){
        a(i)(0) = arr(i);
      }
      a;
    }:Array[Array[Double]])(temp_inst);
    
    println(input.deep);
    println(instraction.deep);

    val rand = new Random;
 		var w1 = DenseMatrix.zeros[Double]( hidden_num, signal_num).map{v => rand.nextDouble()*2 - 1};
    var w2 = DenseMatrix.zeros[Double]( output_num, hidden_num).map{v => rand.nextDouble()*2 - 1};

    for ( loop <- 1 to 10000){
    	for( input_num <- 0 until input.length ){
        
        val in = input(input_num);
        //入力ベクトル
        val x1 = DenseVector(in :+ 1.0); //バイアスベクトルを追加して入力層を生成
    		val x2 = (w1 * x1).map { v => sigmoid(v) }; //順伝播で隠れ層の出力　バイアスベクトルは追加しない
        val x3 = (w2 * x2).map { v => sigmoid(v) }; //順伝播で出力層へ出力
        
        //誤差を計算
        val inst = DenseVector(instraction(input_num)); //教師信号
        val error = (x3 - inst).map{ v => 0.5 * v * v }.sum; 
        //出力層の誤差を計算
        val error_out = (x3-inst) * x3.t * (1.0-x3);        
        //隠れ層の誤差を計算
        val error_hidden = diag(diag(x2*(w2.t*error_out).t) * (1.0 - x2).t);
        
        //重みを更新
        w1 -= (x1 * error_hidden.t).t :* eta;
        w2 -= (x2 * error_out.t).t :* eta;
    	} 
    }
    
    //ここから出力
    def output(in:Array[Double]):Double = {
    		val x1 = DenseVector(in :+ 1.0); //バイアスベクトルを追加して入力層を生成
    		val x2 = (w1 * x1).map { v => sigmoid(v) }; //順伝播で隠れ層の出力　バイアスベクトルは追加しない
    		val x3 = (w2 * x2).map { v => sigmoid(v) }; //順伝播で出力層へ出力
        x3(0);
    }
    val o = input.map{v => output(v)};
    val f = Figure();
    val p = f.subplot(0);
    p += plot(temp_input,temp_inst,'-');
    p += plot(temp_input,o,'+');
  }
  
}