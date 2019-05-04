
import java.util.ArrayList;
import java.util.Random;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.io.BufferedWriter;
import java.io.FileWriter;

public class Test {

  public static void main(String[] args) {

    int input = 56;
    int output =3;
    int hidden =30;
    int width = 3;

    ArrayList<ArrayList<Node>> tData = getArray("test.txt"); // This is the training & test data.

    ArrayList<int[]> tGroup = new ArrayList<>();
    tGroup.add(new int[]{1,2});
    tGroup.add(new int[]{0,2});
    tGroup.add(new int[]{1,0});

    int[] three = {
      0,1,2
    };

    // 0 1 2
    // 1 0 2
    // 2 1 0

    int epochs = 600;

    try {

      for(int a = 0; a < 3; a++) {

        BufferedWriter bTest = new BufferedWriter(new FileWriter("test-output-"+epochs+"-"+three[a]+".csv"));
        BufferedWriter bTrain = new BufferedWriter(new FileWriter("training-output-"+epochs+"-"+tGroup.get(a)[0]+"-"+tGroup.get(a)[1]+".csv"));
        BufferedWriter bWeight = new BufferedWriter(new FileWriter("weights-"+epochs+"-"+tGroup.get(a)[0]+"-"+tGroup.get(a)[1]+".csv"));

        bTrain.write("round,epoch,tGroup,target,outcome,probX,probY,probZ\n");
        bTest.write("round,target,outcome,probX,probY,probZ\n");
        bWeight.write("round,epoch,layer,neuron,type,weight,value\n");

        for(int i = 0; i < 10; i++) {
          NeuralNetwork n = new NeuralNetwork();
          n.buildNetwork(input,hidden,width,output);

          train(bTrain,tData,n,tGroup.get(a),epochs,i);
          test(bTest,tData,n,three[a],epochs,i);

          n.printWeights(bWeight,i,epochs);
        }

        bTest.close();
        bTrain.close();
        bWeight.close();

      }

    } catch(IOException ex) {
      ex.printStackTrace();
    }

  }

  /** Build an ArrayList for 3-fold cross validation. It accepts a file with a series of 8x7 characters separated by spaces.
      The target would be one of three outcomes, since this network is classifying one of three letters. The output layer should
      have three neurons.
  */

  public static ArrayList<ArrayList<Node>> getArray(String filename) {

    ArrayList<ArrayList<Node>> output = new ArrayList<>(3);
    output.add(new ArrayList<>());
    output.add(new ArrayList<>());
    output.add(new ArrayList<>());

    Node n;
    ArrayList<Double> tmp = new ArrayList<>();
    ArrayList<Double> target;
    Random rand = new Random();

    try {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String read;

      while((read = br.readLine())!=null) {

        if(read.length() < 2) {

          if(read.equals("X")) {
            target = new ArrayList<>(Arrays.asList(1.0,0.0,0.0));
            n = new Node(tmp,target,0);
          } else if(read.equals("Y")) {
            target = new ArrayList<>(Arrays.asList(0.0,1.0,0.0));
            n = new Node(tmp,target,1);
          } else {
            target = new ArrayList<>(Arrays.asList(0.0,0.0,1.0));
            n = new Node(tmp,target,2);
          }

          output.get(rand.nextInt(3)).add(n); // Add at random to one of three ArrayLists.
          tmp = new ArrayList<>();

        } else {

          for(int i = 0; i < read.length(); i++) {

            if(Character.compare(read.charAt(i),'.') == 0) {
              tmp.add(0.0);
            } else {
              tmp.add(1.0);
            }

          }

        }

      }

      //System.out.println(output.get(0).size());
      //System.out.println(output.get(1).size());
      //System.out.println(output.get(2).size());

    } catch(IOException ex) {
      ex.printStackTrace();
    }

    return output;

  }


  public static void train(BufferedWriter bw, ArrayList<ArrayList<Node>> a, NeuralNetwork n, int[] tGroup, int epochs, int round) throws IOException {

    Random rand = new Random();
    Node tNode;
    double[] outcome;
    int i = 0;
    int x;

    while(i < epochs) {

      // Train with the two selected groups. Select a random record to train with.
      x = rand.nextInt(tGroup.length);

      tNode = a.get(tGroup[x]).get(rand.nextInt(a.get(tGroup[x]).size()));
      n.forwardPass(tNode.getLetter());
      n.backpropagate(tNode.getLetter(),tNode.getTarget());

      outcome = getOutcome(n);
      bw.write(round+","+i+","+x+","+tNode.getN()+","+(int)outcome[1]+","+n.getLastOutput()+"\n");

      i++;

    }

  }

  public static void test(BufferedWriter bw, ArrayList<ArrayList<Node>> a, NeuralNetwork n, int three, int epochs, int round) throws IOException {

    double[] outcome;
    double correct = 0.0;

    for(Node test : a.get(three)) {

      n.forwardPass(test.getLetter()); // Pass the data through the network.

      outcome = getOutcome(n);

      if((int)outcome[0] == test.getN() && outcome[1] > .60) {
        correct += 1.0;
      } // The outcome needs to be at a sufficient level.

      bw.write(round+","+test.getN()+","+(int)outcome[0]+","+n.getLastOutput()+"\n");

    }

    correct = correct / a.get(three).size();
    System.out.println(correct);

  }

  public static double[] getOutcome(NeuralNetwork n) {

    double[] outcome = new double[2];
    double max = 0.0;
    int maxOutcome = 0;

    for(int c = 0; c < n.getLayers().get(n.getLayers().size()-1).getNeuronCount(); c++) {

      if(n.getLayers().get(n.getLayers().size()-1).getNeurons().get(c).getA() > max) {
        max = n.getLayers().get(n.getLayers().size()-1).getNeurons().get(c).getA();
        maxOutcome = c;
      }

    } // Find the neuron with the highest result. This should match the target of the case.

    outcome[0] = (double)maxOutcome;
    outcome[1] = max;

    return outcome;

  }

  /** Generate an input of a given size, with the added increment */

  public static ArrayList<Double> genInput(int input, int inc) {

    ArrayList<Double> data = new ArrayList<>(input);

    for(int j = 0; j < input; j++) {
      data.add(new Random().nextDouble() + inc);
    }

    return data;

  }

}
