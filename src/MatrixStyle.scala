import scala.util.Random;
import breeze.linalg._;
//import scalala.scalar._;

object MatrixStyle {

  def sigmoid(v:Double):Double = {
    1 / ( 1 + Math.exp(-v));
  }
  
  def main(args: Array[String]): Unit = {
    val signal_num = 2; //バイアス含めた入力次元数
    val hidden_num = 4; //隠れ層の次元数
    val output_num = 1;
    
    val N = 5;
    val temp = ( -1.0 to 1.0 by 2.0 / N ).toArray;
    val input = ((arr:Array[Double]) => {
      	var a = Array.ofDim[Double](temp.length, signal_num-1);
      	for( i <- 0 until temp.length){
      		a(i)(0) = temp(i);
      	}
      	a;
      }:Array[Array[Double]])(temp);
    val instraction = Array.ofDim[Double](temp.length, output_num);

    val rand = new Random;
 		var w1 = DenseMatrix.zeros[Double]( hidden_num, signal_num).map{v => rand.nextDouble()*2 - 1};
    var w2 = DenseMatrix.zeros[Double]( output_num, hidden_num).map{v => rand.nextDouble()*2 - 1};

    for ( loop <- 1 to 2){
    	for( in <- input ){
        //入力ベクトル
    		val x1 = ((arr:Array[Double]) => {
          DenseVector(arr :+ 1.0);//バイアスベクトルを追加して行列を作成
          /*
              var a = DenseVector.zeros[Double]( signal_num );
              for(b <- 0 until arr.length) {
                a(b) = arr(b);
              }
              a(a.length-1) = 1.0; //バイアスベクトルを追加
              a;
           */
            }:DenseVector[Double])(in);

    		val x2 = (w1 * x1).map { v => sigmoid(v) };
        println(x2);
        //バイアスベクトルは追加しない
        val x3 = (w2 * x2).map { v => sigmoid(v) };
        println(x3);
    	}
    }
  }
}