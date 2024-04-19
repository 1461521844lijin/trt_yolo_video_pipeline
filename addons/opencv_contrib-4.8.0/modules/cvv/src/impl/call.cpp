#include "call.hpp"
#include <atomic>

namespace cvv
{
namespace impl
{

size_t newCallId()
{
	static std::atomic_size_t nextId (1);
	return nextId++;
}

Call::Call() : metaData_{}, id{ newCallId() }, calltype{}
{
}

Call::Call(impl::CallMetaData callData, QString type, QString description,
           QString requestedView)
    : metaData_{ std::move(callData) }, id{ newCallId() },
      calltype{ std::move(type) }, description_{ std::move(description) },
      requestedView_{ std::move(requestedView) }
{

}
}
} // namespaces cvv::impl
