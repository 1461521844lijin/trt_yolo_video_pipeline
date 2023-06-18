#pragma once
#include "base/ProcessNode.hpp"
#include "base/TransferData.h"

#include "xencode.h"
#include "xmux.h"
#include "xscale.h"

namespace FFmpeg
{


    class StreamPusher: public Base::Node
    {
    private:
        XMux    m_mux;
        XEncode m_encode;
        XScale  m_scale;

        int from_width = 0;
        int from_height = 0;
        int from_format = 0;
        int to_width = 0;
        int to_height = 0;
        int to_format = 0;
       
    public:
        StreamPusher(std::string name):Node(name){}
        ~StreamPusher() = default;

        void set_frommat(int width, int height, int format);
        void set_tomat(int width, int height, int format);

        virtual void worker() override;

        void Open(const std::string url, bool useHwEncode = false);


    };
    

    
    
} // namespace FFmpeg



