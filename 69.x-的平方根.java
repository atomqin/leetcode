/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-31 17:29:59
 * @LastEditTime: 2021-04-10 09:56:32
 */
/*
 * @lc app=leetcode.cn id=69 lang=java
 *
 * [69] x 的平方根
 */

// @lc code=start
class Solution {
    public int mySqrt(int x) {
        if (x == 0)
            return x;
        if(x == 1 || x == 2 || x== 3)
            return 1;
        int left = 1, right = x/2;
        while (left < right) {
            int mid = (left + right) >>> 1;
            int sqrt = x / mid;
            if (mid < sqrt) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        //不要用 right * right > x 判断，会溢出
        if (right > x / right)
            return right - 1;
        return right;
    }
}
// @lc code=end

