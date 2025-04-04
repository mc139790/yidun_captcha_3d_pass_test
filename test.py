import re
import cv2
import numpy as np
import requests
from playwright.async_api import Page, async_playwright
import asyncio

type_dict = {
    'color_only': re.compile(r"^请点击([红绿蓝黄灰])色(.{1,3})$"),
    'orientation_only': re.compile(r"^请点击([正侧])向的(.{1,3})$"),
    'same_orientation': re.compile(r"^请点击(.{1,3})朝向一样的(.{1,3})$"),
    'same_color': re.compile(r"^请点击(.{1,3})颜色一样的(.{1,3})$"),
    'color_than_same_orientation': re.compile(r"^请点击([红绿蓝黄灰])色(.{1,3})朝向一样的(.{1,3})$"),
}


# HSV颜色范围字典
color_dict = {
    '红': [([0, 100, 132], [10, 255, 255]), ([160, 100, 132], [180, 255, 255])],
    '绿': [([40, 100, 132], [80, 255, 255])],
    '蓝': [([100, 100, 132], [140, 255, 255])],
    '黄': [([20, 100, 132], [40, 255, 255])],
    '灰': [([0, 0, 0], [180, 64, 192])],
}


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


async def click(page: Page, i, j):
    box = await page.locator('div.yidun_panel-placeholder').bounding_box()
    await page.mouse.click(int(box['x'] + j), int(box['y'] + i))

async def progress_captcha(page: Page):
    while True:
        await page.wait_for_timeout(500)
        # 等待滑块验证码图片加载完成
        if await page.locator("div.yidun--loading").is_visible():
            await page.wait_for_selector("div.yidun--loading", state="detached")

        img_url = await page.locator('img.yidun_bg-img').get_attribute('src')
        text = await page.locator('span.yidun_tips__text').inner_text()
        print(text)

        img = await download_image(img_url)
        text_type, match_groups = await text_template_case(text)
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
            cv2.imwrite('sobelx.png', sobelx)
            sobelx_bin = cv2.inRange(sobelx, 32, 255)
            cv2.imwrite('sobelx_bin.png', sobelx_bin)
            mask = cv2.bitwise_and(mask, sobelx_bin)
        cv2.imwrite('mask.png', mask)
        cv2.blur(mask, (9, 9), mask)
        cv2.imwrite('blur_mask.png', mask)
        mask = await non_maximum_suppression(mask)
        cv2.imwrite('non_maximum_suppression_mask.png', mask)
        result = await get_maximum_point(mask)
        if result is None:
            return False
        i, j = result
        await click(page, i, j)
        return True
    else:
        return False


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
