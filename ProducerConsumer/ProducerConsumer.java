import java.util.Random;

public class ProducerConsumer {
   public static void main(String[] args) {
      	BoundedBuffer bb = new BoundedBuffer();
      	Producer prod = new Producer(bb);
      	Consumer con = new Consumer(bb);
      	prod.start(); 
      	con.start();
        System.out.println("Finished");
	
}
}
class BoundedBuffer {
   int Max_size = 1000;
   int size=0;
   private Double val;
   public synchronized Double get() { 

      while (size == 0) {
         //System.out.println(size);
         try {wait();}
         catch (InterruptedException e) {}
      }
      size = size -1;
      notifyAll();
      return val;
      }
   public synchronized void put(Double bufferElement) {
      while (size == Max_size) {
         try {wait();}
         catch (InterruptedException e) {} 
      }
      size = size+1;
      val = bufferElement;
      notifyAll();
      }
}

class Consumer extends Thread {
   double bufferValueCounter = 0;
   private BoundedBuffer bb;
   int numberofcon = 0;

   public Consumer(BoundedBuffer boundb) {
      bb = boundb;
   }

   public void run() {
      Double value = (double) 0;
         for (int i = 0; i < 1000001; i++) {
            value = bb.get();
            bufferValueCounter = bufferValueCounter + value;
            if (i%100000 == 0){
               if( i != 0){
               System.out.println("Consumer: Consumed " + numberofcon  +" items, Cumulative value of consumed items =" +bufferValueCounter);
                  }
	    }
            numberofcon = numberofcon + 1;
            }

}
}

class Producer extends Thread {

	double bufferValueCounter = 0;
	int numberofpro = 0;
	private BoundedBuffer bb;


	public Producer(BoundedBuffer boundb) {
		bb = boundb;

	}

	public void run() {
	        Random random = new Random();
		for (int i = 0; i < 1000001; i++) {
			Double bufferElement = random.nextDouble() * 100.0;
    			bb.put(bufferElement);
    			bufferValueCounter = bufferValueCounter+bufferElement;
                        if (i%100000 == 0){
              			 if( i != 0){
                             System.out.println("Producer: Generated " +numberofpro+ " items, Cumulative value of generated items= " +bufferValueCounter);

                                }
                        }

    			numberofpro = numberofpro +1;

}

}
}

