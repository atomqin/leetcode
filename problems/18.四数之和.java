import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-30 21:59:02
 * @LastEditTime: 2021-03-30 22:33:08
 */
/*
 * @lc app=leetcode.cn id=18 lang=java
 *
 * [18] 四数之和
 */

// @lc code=start
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length < 4)
            return res;
        int len = nums.length;
        Arrays.sort(nums);
        //定义4个指针i,j,lo,hi从0开始遍历，j从i+1开始遍历，留下lo和hi，lo指向j+1，hi指向数组最大值
        for (int i = 0; i < len - 3; i++) {
            //去重
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            //若最小值比target大，后面不用继续了
            int min1 = nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3];
            if (min1 > target)
                break;
            //当前循环的最大值比target小，本轮循环跳过
            int max1 = nums[i] + nums[len - 1] + nums[len - 2] + nums[len - 3];
            if (max1 < target)
                continue;
            // 第二层循环
            for (int j = i + 1; j < len - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1])
                    continue;
                int lo = j + 1, hi = len - 1;
                int min2 = nums[i] + nums[j] + nums[lo] + nums[lo + 1];
                if (min2 > target)
                    break;
                int max2 = nums[i] + nums[j] + nums[hi] + nums[hi - 1];
                if (max2 < target)
                    continue;
                while (lo < hi) {
                    int curr = nums[i] + nums[j] + nums[lo] + nums[hi];
                    if (curr == target) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[lo], nums[hi]));
                        lo++;
                        while (lo < hi && nums[lo] == nums[lo - 1])
                            lo++;
                        hi--;
                        while (lo < hi && nums[hi] == nums[hi + 1])
                            hi--;
                        // while (lo < hi && j < hi && nums[hi] == nums[hi - 1])
                        //     hi--;
                        // hi--;
                    } else if (curr < target)
                        lo++;
                    else
                        hi--;
                }
            }
        }
        return res;
    }
}
// @lc code=end
