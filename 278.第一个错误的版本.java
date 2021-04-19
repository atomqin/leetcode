/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-10 15:04:59
 * @LastEditTime: 2021-04-10 15:13:01
 */
/*
 * @lc app=leetcode.cn id=278 lang=java
 *
 * [278] 第一个错误的版本
 */

// @lc code=start
/* The isBadVersion API is defined in the parent class VersionControl.
      boolean isBadVersion(int version); */

public class Solution extends VersionControl {
    public int firstBadVersion(int n) {
        int l = 1,r = n;
        while (l < r) {
            int m = (l + r) >>> 1;
            if (isBadVersion(m)) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }
}
// @lc code=end

