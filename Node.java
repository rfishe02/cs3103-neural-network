
import java.util.ArrayList;

public class Node {

  private ArrayList<Double> letter;
  private ArrayList<Double> target;
  private int n;

  public Node(ArrayList<Double> letter, ArrayList<Double> target, int n) {
    this.letter = letter;
    this.target = target;
    this.n = n;
  }

  public ArrayList<Double> getLetter() {
    return letter;
  }

  public ArrayList<Double> getTarget() {
    return target;
  }

  public int getN() {
    return n;
  }

}
