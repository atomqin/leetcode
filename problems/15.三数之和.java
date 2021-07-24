import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-30 20:15:04
 * @LastEditTime: 2021-03-31 09:11:14
 */
/*
 * @lc app=leetcode.cn id=15 lang=java
 *
 * [15] 三数之和
 */

// @lc code=start
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3 || nums == null)
            return res;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            int lo = i + 1, hi = nums.length - 1;
            if (nums[i] > 0)
                return res;
            int min = nums[i] + nums[lo] + nums[lo + 1];
            if (min > 0)
                break;
            int max = nums[i] + nums[hi] + nums[hi - 1];
            if (max < 0)
                continue;
            //第一个元素不能重复
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            while (lo < hi) {
                if (nums[lo] + nums[hi] == -nums[i]) {
                    res.add(Arrays.asList(nums[i], nums[lo], nums[hi]));
                    while (lo < hi && nums[lo] == nums[lo + 1])//去重
                        lo++;
                    while (lo < hi && nums[hi] == nums[hi - 1])//去重
                        hi--;
                    lo++;
                    hi--;
                } else if (nums[lo] + nums[hi] < -nums[i])
                    lo++;
                else
                    hi--;
            }
        }
        return res;
    }
}
// @lc code=end

