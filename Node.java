
import java.util.ArrayList;

public class Node {

  private ArrayList<Double> letter;
  private ArrayList<Double> target;

  public Node(ArrayList<Double> letter, ArrayList<Double> target) {
    this.letter = letter;
    this.target = target;
  }

  public ArrayList<Double> getLetter() {
    return letter;
  }

  public ArrayList<Double> getTarget() {
    return target;
  }

  public void setLetter(ArrayList<Double> letter) {
    this.letter = letter;
  }

  public void setTarget(ArrayList<Double> target) {
    this.target = target;
  }

}
