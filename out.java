import java.io.*;
import java.util.*;
 
public class out
{	
    public static void main(String[] args)
    {
		function obj = new function();
      
		List<Integer> sum_data = new ArrayList<Integer>();
	  
		sum_data.add(1);
		sum_data.add(2);
		sum_data.add(3);
		sum_data.add(4);
		sum_data.add(5);

		
		obj.sum(sum_data);
		
    }
    
}

class function{
	

	public void sum(List<Integer> list)
	{
		int sum = 0;
		for(int i=0; i<list.size(); i++){
				sum+=list.get(i);
			}
		System.out.println("sum is "+sum);
	}
}