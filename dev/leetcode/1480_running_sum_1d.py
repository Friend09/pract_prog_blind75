def running_sum(nums):
    running_total = 0
    result = []
    for num in nums:
        running_total += num
        result.append(running_total)
    return result


nums = [3, 1, 2, 10, 1]
print(running_sum(nums))
