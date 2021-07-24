import java.util.Arrays;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-30 21:12:09
 * @LastEditTime: 2021-03-30 21:40:10
 */
/*
 * @lc app=leetcode.cn id=16 lang=java
 *
 * [16] 最接近的三数之和
 */

// @lc code=start
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            int lo = i + 1, hi = nums.length - 1;
            while (lo < hi) {
                int sum = nums[i] + nums[lo] + nums[hi];
                if (sum == target)
                    return sum;
                if (Math.abs(sum - target) < Math.abs(res - target)) {
                    res = sum;
                } else if (nums[i] + nums[lo] + nums[hi] > target)
                    hi--;
                else
                    lo++;
            }
        }
        return res;
    }
}
// @lc code=end
