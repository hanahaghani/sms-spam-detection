# SMS Spam Classification

این ریپازیتوری شامل دو رویکرد برای تشخیص پیام‌های Spam و Ham است:

- Machine Learning: استفاده از TF-IDF و مدل‌های کلاسیک یادگیری ماشین
- Deep Learning: استفاده از TF-IDF و یک شبکه عصبی سفارشی با PyTorch

## Deep Learning

در این بخش، هدف علاوه بر ساخت مدل، یادگیری عملی فرآیند آموزش و بهینه‌سازی شبکه عصبی، کنترل Overfitting و بررسی تأثیر تکنیک‌های مختلف بود.

ویژگی‌های متنی با TF-IDF به 1056 ویژگی عددی تبدیل شدند و به شبکه‌ای با معماری زیر داده شدند:

1056 → 256 → 128 → 64 → 1

برای آموزش و بهبود مدل از موارد زیر استفاده و آزمایش شد:

- ReLU و Leaky ReLU
- Dropout و Batch Normalization
- Adam، Adamax و AdamW
- Weight Decay
- StepLR برای کاهش تدریجی Learning Rate
- Early Stopping برای کنترل Overfitting
- BCEWithLogitsLoss برای Binary Classification
- DataLoader با Batch Size برابر 32

در نهایت، ترکیب ReLU + Dropout + Adam + Weight Decay + StepLR + Early Stopping به عنوان تنظیم نهایی انتخاب شد.

مدل خروجی را به صورت Logit تولید می‌کند و با استفاده از Sigmoid و Threshold برابر 0.5، پیام به یکی از دو کلاس Ham یا Spam اختصاص داده می‌شود.

همچنین یک Visualization برای نمایش عبور یک پیام جدید از مراحل TF-IDF و لایه‌های شبکه و مشاهده Activationها و احتمال Spam بودن پیام ایجاد شده است.

## Machine Learning

در نسخه Machine Learning، از TfidfVectorizer و Logistic Regression برای تشخیص Spam استفاده شد.

این مدل به دقت حدود 98٪ رسید و با تنظیم Decision Threshold، حساسیت تشخیص پیام‌های Spam بهبود داده شد.

مسیر توسعه NLP

این ریپازیتوری بخشی از مسیر یادگیری من در حوزه NLP و Machine Learning است.

پروژه‌های آینده شامل:

- تحلیل احساسات فارسی: استفاده از ابزارهایی مانند Hazm برای پردازش متن فارسی
- تحلیل متن چندزبانه: توسعه مدل‌هایی برای تحلیل متن در زبان‌هایی مانند فارسی، انگلیسی، عربی و ترکی

هدف این مسیر، تقویت مهارت عملی در NLP و همچنین کار بیشتر روی پردازش زبان فارسی است.

# Technologies

Python · Pandas · NumPy · Scikit-learn · PyTorch · Matplotlib · TF-IDF 