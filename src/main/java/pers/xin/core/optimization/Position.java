package pers.xin.core.optimization;


import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by xin on 26/03/2018.
 */
public class Position implements Serializable {
    private static final long serialVersionUID = 8177502680557477881L;
    private double[] dimension;

    public Position(double... dimension) {
        this.dimension = dimension.clone();
    }

    public Position(Position position){
        this.dimension = position.dimension.clone();
    }

    public static final Position NONE = new Position();

    @Override
    public boolean equals(Object obj) {
        if(this==obj)return true;
        if( !(obj instanceof Position)) return false;
        Position x = (Position) obj;
        if(dimension.length!=x.dimension.length) return false;
        for (int i = 0; i < this.dimension.length; i++) {
            if( x.dimension[i]!=this.dimension[i])return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int value = 0;
        for (int i = 0; i < dimension.length; i++) {
            value += dimension[i];
        }
        return value;
    }

    public double[] get() {
        return dimension.clone();
    }

    @Override
    public String toString() {
        return Arrays.toString(dimension);
    }
}
