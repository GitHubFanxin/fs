package pers.xin.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by xin on 15/03/2018.
 */
public class Output {
    public static final String time = (new SimpleDateFormat("yyMMddhhmm")).format(new Date());
    private static String folder = "./output/"+time+"/";

    public static void setFolder(String folder) {
        Output.folder = folder;
    }

    public static PrintWriter createAppendPrint(String name) throws Exception{
        File file = getFile(folder +name+".csv");
        PrintWriter pw = new PrintWriter(new FileWriter(file,true));
        return pw;
    }

    public static PrintWriter createPrint(String name) throws Exception{
        File file = getFile(folder +name+".csv");
        PrintWriter pw = new PrintWriter(new FileWriter(file));
        return pw;
    }

    private static File getFile(String name) throws Exception{
        File file = new File(name);
        if(!file.getParentFile().exists()){
            file.getParentFile().mkdirs();
        }
        if(!file.exists()){
            file.createNewFile();
        }
        return file;
    }
}
