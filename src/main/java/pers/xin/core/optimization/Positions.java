package pers.xin.core.optimization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by xin on 30/03/2018.
 */
public class Positions implements Serializable{
    private static final long serialVersionUID = 1074191679350134852L;
    private HashMap<Position,Double> map = new HashMap<>();
    private HashMap<Position,Integer> markTimes = new HashMap<>();
    private Position bestPosition = Position.NONE;
    public HashMap<Position,ArrayList<Double>> history = new HashMap<>();

    public Positions() {
        map.put(bestPosition,Double.NEGATIVE_INFINITY);
    }

    public Double mark(Position position, Double value){
        position = new Position(position);
        if(map.containsKey(position)){//if current position has been visited
            int markTime = markTimes.get(position);
            double average = updateFunction(map.get(position),markTime,value);
            markTimes.put(position,markTime+1);
            map.put(position,average);
            history.get(position).add(value);
            if(position.equals(bestPosition)){//if current position is the best position update best position
                Double max = Double.NEGATIVE_INFINITY;
                Position tmp = bestPosition;
                for (Map.Entry<Position, Double> positionDoubleEntry : map.entrySet()) {
                    if(positionDoubleEntry.getValue()>max){
                        tmp = positionDoubleEntry.getKey();
                        max = positionDoubleEntry.getValue();
                    }
                }
                if(!bestPosition.equals(tmp)) {
                    bestPosition = tmp;
                }
            }else {
                if(average > map.get(bestPosition)){//if current position fitness is bigger than best position
                    bestPosition = position;
                }
            }
            return average;
        }else {
            markTimes.put(position,1);
            map.put(position,value);
            ArrayList<Double> h = new ArrayList<>();
            h.add(value);
            history.put(position,h);
            if(value >= map.get(bestPosition)){ //avoid the bug of none gbest for next move
                bestPosition = position;
            }
            return value;
        }
    }

    public Double get(Position position){
        return map.get(position);
    }

    public Position getBestPosition() {
        return new Position(bestPosition);
    }

    public Double getBestValue(){
        return map.get(bestPosition);
    }

    public HashMap<Position, ArrayList<Double>> getHistory() {
        return history;
    }

    @Override
    public String toString() {
        return "Positions{" +
                "bestPosition=" + bestPosition +
                '}';
    }

    private double updateFunction(double a,int time, double b){
        return (a*time+b)/(time+1);
    }

    public String getHistory(Object key){
        if(!history.containsKey(key)) return "no history";
        return history.get(key).stream().map(String::valueOf).collect(Collectors.joining(", "));
    }
}
