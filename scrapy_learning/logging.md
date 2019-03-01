
```python
import logging
……
	logging.warning(item)
……
```
**显示:**   
`[time] [root] WARNING item`

不能显示logging的来源，所以使用：

```python
import logging
logger = logging.getLogger(__name__)	
# 实例化，其他文件中可from **.py import logger
……
	logger.warning(item)
……
```

**显示:**
`[time] [location] WARNING item`

>### 其他
* settings.py中设置LOG_LEVEL = “WARNING”表示只显示WARNING及以上等级的日志
* settings.py中设置LOG_FILE = “./a.log”，使日志存储在根目录的a.log中，不显示在屏幕。
* 新建一个log.py，可使用logging.basicConfig()设置日志格式，可百度参考格式。
