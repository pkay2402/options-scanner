# 🔐 Schwab API Authentication Guide

## Step-by-Step Instructions

### 1. Open Authorization URL
Visit this URL in your browser:

```
https://api.schwabapi.com/oauth/authorize?response_type=code&client_id=RCAoyI2iLVrtlVyS3BFDzKNQ7Hw6JaRk&redirect_uri=https%3A%2F%2F127.0.0.1%3A3000%2Fcallback&scope=readonly
```

### 2. Login and Authorize
- Log in to your Schwab account
- Review the permissions requested
- Click "Authorize" or "Allow"

### 3. Get the Authorization Code
After authorizing, you'll be redirected to a URL that looks like:
```
https://127.0.0.1:3000/callback?code=ABC123XYZ789...
```

**The page will show an error - that's normal!**

### 4. Copy the Code
From the redirect URL, copy ONLY the value after `code=`

For example, if the URL is:
```
https://127.0.0.1:3000/callback?code=ABC123XYZ789&state=something
```

Then your authorization code is: `ABC123XYZ789`

### 5. Update Your Configuration
Once you have the authorization code, run:

```bash
python3 scripts/auth_setup.py
```

And paste your authorization code when prompted.

### 6. Test Authentication
After setting up, test the connection:

```bash
python3 scripts/test_auth.py
```

---

## Important Notes

- ⏱️ Authorization codes expire quickly (usually within 10 minutes)
- 🔄 You may need to repeat this process if the code expires
- 🚫 The redirect page showing an error is expected behavior
- 📋 Only copy the code parameter value, not the entire URL

## Need Help?

If you run into issues:
1. Make sure you're logged into the correct Schwab account
2. Check that your Client ID and Secret are correct
3. Try generating a new authorization code
4. Ensure you're copying only the code value, not the full URL