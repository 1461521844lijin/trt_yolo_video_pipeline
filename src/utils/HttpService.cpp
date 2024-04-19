#include "HttpService.hpp"

HttpService::HttpService(){}
HttpService::~HttpService(){}


size_t http_data_writer(void* data, size_t size, size_t nmemb, void* content)
{
	long totalSize = size * nmemb;
	//强制转换
	std::string* symbolBuffer = (std::string*)content;
	if (symbolBuffer)
	{
		symbolBuffer->append((char *)data, ((char*)data) + totalSize);
		return totalSize;
	};
	//cout << "symbolBuffer:" << symbolBuffer << endl;
	//返回接受数据的多少
	return totalSize;
}

bool HttpService::is_base64(const char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}


std::string HttpService::base64_encode(const unsigned char * bytes_to_encode, unsigned int in_len)
{
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
 
    while (in_len--)
    {
        char_array_3[i++] = *(bytes_to_encode++);
        if(i == 3)
        {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;     
            for(i = 0; (i <4) ; i++)
            {
                ret += m_base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }
    if(i)
    {
        for(j = i; j < 3; j++)
        {
            char_array_3[j] = '\0';
        }
 
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
 
        for(j = 0; (j < i + 1); j++)
        {
            ret += m_base64_chars[char_array_4[j]];
        }
 
        while((i++ < 3))
        {
            ret += '=';
        }
 
    }
    return ret;
}


std::string HttpService::base64_decode(std::string const & encoded_string)
{
    int in_len = (int) encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;
 
    while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i ==4) {
            for (i = 0; i <4; i++)
                char_array_4[i] = m_base64_chars.find(char_array_4[i]);
 
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
 
            for (i = 0; (i < 3); i++)
                ret += char_array_3[i];
            i = 0;
        }
    }
    if (i) {
        for (j = i; j <4; j++)
            char_array_4[j] = 0;
 
        for (j = 0; j <4; j++)
            char_array_4[j] = m_base64_chars.find(char_array_4[j]);
 
        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);  
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];  
 
        for (j = 0; (j < i - 1); j++) ret += char_array_3[j];  
    }  
 
    return ret;  
}

std::string HttpService::b64_encode(cv::Mat img)
{

    m_vec_compresion.push_back(cv::IMWRITE_JPEG_QUALITY);
    m_vec_compresion.push_back(90);

    cv::imencode(".jpg", img, m_vec_img, m_vec_compresion);
    std::string b64_str = base64_encode(m_vec_img.data(), m_vec_img.size());
    return b64_str;
}


std::string HttpService::requesthttp(std::string sendstr, std::string url)
{
	std::string str_data;
	CURL *p_curl = NULL;
	CURLcode res;
	p_curl = curl_easy_init();
    //logger = spdlog::get("logger");

	if (NULL != p_curl)
	{
		// Set timeout -> 1 sec
		curl_easy_setopt(p_curl, CURLOPT_TIMEOUT, 1);

		// Set url
		curl_easy_setopt(p_curl, CURLOPT_URL, const_cast<char*>(url.c_str()));

		// Set http send str -> json
		curl_slist* plist = curl_slist_append(NULL,
			"Content-Type:application/json;charset=UTF-8");
		curl_easy_setopt(p_curl, CURLOPT_HTTPHEADER, plist);

		// Set post json
		curl_easy_setopt(p_curl, CURLOPT_POSTFIELDS, sendstr.c_str());

		// Set callback
		curl_easy_setopt(p_curl, CURLOPT_WRITEFUNCTION, http_data_writer);

		// Set write data
		curl_easy_setopt(p_curl, CURLOPT_WRITEDATA, (void*)&str_data);

		// post
		res = curl_easy_perform(p_curl);
		curl_easy_cleanup(p_curl);

		// check for error
		if (res!=CURLE_OK)
		{
			return "error";
		}
		curl_global_cleanup();
	}
	return "ok";
}