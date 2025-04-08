import re
import cv2
import numpy as np
import requests
from playwright.async_api import Page, async_playwright
from make_template import make_template
import asyncio

type_dict = {
    'color_only': re.compile(r"^请点击([红绿蓝黄灰])色(.{1,3})$"),
    'orientation_only': re.compile(r"^请点击([正侧])向的(.{1,3})$"),
    'same_orientation': re.compile(r"^请点击(.{1,3})朝向一样的(.{1,3})$"),
    'same_color': re.compile(r"^请点击(.{1,3})颜色一样的(.{1,3})$"),
    'color_than_same_orientation': re.compile(r"^请点击([红绿蓝黄灰])色(.{1,3})朝向一样的(.{1,3})$"),
    'refresh': re.compile("失败过多，点此重试"),
}


# HSV颜色范围字典
color_dict = {
    '红': [([0, 100, 132], [10, 255, 255]), ([160, 100, 132], [180, 255, 255])],
    '绿': [([40, 100, 132], [80, 255, 255])],
    '蓝': [([100, 100, 132], [140, 255, 255])],
    '黄': [([20, 100, 132], [40, 255, 255])],
    '灰': [([0, 0, 0], [180, 64, 192])],
}

support_text = [
    '小写a', '小写b', '小写c', '小写d', '小写e', '小写f', '小写g', '小写h',
    '小写i', '小写j', '小写k', '小写l', '小写m', '小写n', '小写o', '小写p',
    '小写q', '小写r', '小写s', '小写t', '小写u', '小写v', '小写w', '小写x',
    '小写y', '小写z',
    '大写A', '大写B', '大写C', '大写D', '大写E', '大写F', '大写G', '大写H',
    '大写I', '大写J', '大写K', '大写L', '大写M', '大写N', '大写O', '大写P',
    '大写Q', '大写R', '大写S', '大写T', '大写U', '大写V', '大写W', '大写X',
    '大写Y', '大写Z',
    '数字0', '数字1', '数字2', '数字3', '数字4', '数字5', '数字6', '数字7',
    '数字8', '数字9',
]

# 几乎无法识别明亮物体
def template_match(gray_img, text, is_forward):
    char = text[-1]
    is_blk = False if char.islower() else True
    gray_img = gray_img.copy()
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    img_sobel_x = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, ksize=3)
    img_sobel_y = cv2.Sobel(gray_img, cv2.CV_8U, 0, 1, ksize=3)
    img_sobel = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)

    source_template = make_template(char, False, is_blk) # 软渲染很慢，可以考虑使用硬件加速
    template_size_list = [(69, 69), (66, 66), (63, 63), (60, 60), (57, 57), (54, 54), (51, 51), (48, 48), (45, 45)]
    max_value = 0
    max_point = None
    for template_size in template_size_list:
        template = cv2.resize(source_template, template_size, interpolation=cv2.INTER_CUBIC)
        template = cv2.GaussianBlur(template, (3, 3), 0)
        template_sobel_x = cv2.Sobel(template, cv2.CV_8U, 1, 0, ksize=3)
        template_sobel_y = cv2.Sobel(template, cv2.CV_8U, 0, 1, ksize=3)
        template = cv2.addWeighted(template_sobel_x, 0.5, template_sobel_y, 0.5, 0)
        result = cv2.matchTemplate(img_sobel, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= max_value:
            max_value = max_val
            max_point = [max_loc[0] + template_size[0] // 2, max_loc[1] + template_size[1] // 2]
        
    return max_point

async def non_maximum_suppression(mask):
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    return eroded_mask


async def get_maximum_point(mask):
    max_point = None
    max_value = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > max_value:
                max_value = mask[i, j]
                max_point = (i, j)
    return max_point


async def download_image(url):
    response = requests.get(url)
    # 转换为numpy数组供cv2使用
    image_array = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


async def text_template_case(text):
    for key, pattern in type_dict.items():
        match = pattern.match(text)
        if match:
            return key, match.groups()
    pass # TODO: 处理未知的验证码提示文本


async def click(page: Page, x, y):
    box = await page.locator('div.yidun_panel-placeholder').bounding_box()
    await page.mouse.click(int(box['x'] + x), int(box['y'] + y))

async def progress_captcha(page: Page):
    while True:
        await page.wait_for_timeout(500)
        # 等待验证码图片加载完成
        if await page.locator("div.yidun--loading").is_visible():
            await page.wait_for_selector("div.yidun--loading", state="detached")

        img_url = await page.locator('img.yidun_bg-img').get_attribute('src')
        text = await page.locator('span.yidun_tips__text').inner_text()
        print(text)

        img = await download_image(img_url)
        text_type, match_groups = await text_template_case(text)
        if text_type == 'refresh':
            await page.locator('button.yidun_refresh').click()
            await page.wait_for_timeout(500)
            continue
        cv2.imwrite('img.png', img)

        if await progress_img(page, img, text_type, match_groups):
            await page.wait_for_timeout(1000)
            if await page.locator('div.yidun_modal__body').is_hidden():
                break
            else:
                await page.locator('button.yidun_refresh').click()
                await page.wait_for_timeout(500)
        else:
            await page.locator('button.yidun_refresh').click()
            await page.wait_for_timeout(500)

async def progress_img(page: Page, img, text_type, match_groups):
    if text_type == 'color_only':
        color, _ = match_groups
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_ranges = color_dict[color]
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask += cv2.inRange(hsv_img, lower, upper)
        if color == '灰':
            sobelx = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_8U, 1, 0, ksize=3)
            sobelx_bin = cv2.inRange(sobelx, 32, 255)
            mask = cv2.bitwise_and(mask, sobelx_bin)
        cv2.blur(mask, (9, 9), mask)
        mask = await non_maximum_suppression(mask)
        result = await get_maximum_point(mask)
        if result is None:
            return False
        y, x = result
        await click(page, x, y)
        return True
    elif text_type == 'orientation_only':
        orientation, char = match_groups
        is_forward = True if orientation == '正' else False
        point = template_match(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), char, is_forward)
        await click(page, point[0], point[1])
        return True
    else:
        char = match_groups[-1]
        is_forward = False
        point = template_match(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), char, is_forward)
        await click(page, point[0], point[1])
        return True


async def captcha_3d_pass():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://dun.163.com/trial/space-inference")
        await page.wait_for_timeout(1000)
        await (await page.query_selector_all("li.tcapt-tabs__tab"))[2].click()
        await page.wait_for_timeout(500)
        await page.locator('button.tcapt-bind_btn').click()
        await progress_captcha(page)
        await page.wait_for_timeout(60000)
        await browser.close()
    

asyncio.run(captcha_3d_pass())
