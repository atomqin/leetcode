import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Map;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-25 20:06:01
 * @LastEditTime: 2021-03-26 19:01:58
 */
/*
 * @lc app=leetcode.cn id=1 lang=java
 *
 * [1] 两数之和
 */

// @lc code=start
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[] { map.get(target - nums[i]), i };
            } else
                map.put(nums[i], i);
        }
        return new int[2];
    }
}
// @lc code=end

