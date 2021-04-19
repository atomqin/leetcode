import java.awt.List;
import java.util.Comparator;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-17 09:19:49
 * @LastEditTime: 2021-04-01 19:46:10
 */
/*
 * @lc app=leetcode.cn id=406 lang=java
 *
 * [406] 根据身高重建队列
 */

// @lc code=start
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        
        //身高降序，个数升序，将某个学生插入到第k个位置
         Arrays.sort(people, (o1, o2) -> (o1[0] == o2[0]) ? o1[1] - o2[1] : o2[0] - o1[0]);
         List<int[]> list = new ArrayList<>();
         for (int[] i : people) {
             list.add(i[1], i);
         }
         return list.toArray(new int[list.size()][]);
    }
}
// @lc code=end

