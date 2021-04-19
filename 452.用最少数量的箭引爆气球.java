/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-16 16:48:11
 * @LastEditTime: 2021-03-16 22:23:59
 */
/*
 * @lc app=leetcode.cn id=452 lang=java
 *
 * [452] 用最少数量的箭引爆气球
 */

// @lc code=start
class Solution {
    public int findMinArrowShots(int[][] points) {
        if (points.length == 0)
            return 0;
        /* Arrays.sort(points, new Comparator<int[]>(){
            public int compare(int[] o1, int[] o2){
                return (o1[0] < o2[0]) ? -1 : ((o1[0] == o2[0]) ? 0 : 1);
            }
        }); */
        // Arrays.sort(points,Comparator.comparingInt(o -> o[1]));
        Arrays.sort(points,(o1,o2) -> (o1[1] < o2[1]) ? -1 : 1);
        int cnt = 1, end = points[0][1];
        for (int[] point : points) {
            // int start = point[0];
            if (point[0] > end) {
                cnt++;
                end = point[1];
            }
        } 
        return cnt;
    }
}
// @lc code=end

