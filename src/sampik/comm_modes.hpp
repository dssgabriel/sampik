/**
 * Copyright (C) 2024, CEA
 *
 * Licensed under the MIT License ("the License" hereafter);
 * You may not use this file except in compliance with the License.
 * You should have received a full copy of the License along with this program;
 * if not, you can obtain a copy at:
 *
 *    https://opensource.org/license/MIT
 *
 * The software is provided “as is”, without warranty of any kind, express
 * or implied, including but not limited to the warranties of merchantability,
 * fitness for a particular purpose and noninfringement. In no event shall
 * the authors or copyright holders be liable for any claim, damages or other
 * liability, whether in an action of contract, tort or otherwise, arising from,
 * out of or in connection with the software or the use or other dealings
 * in the software.
 *
 * Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr>
 **/

#pragma once

#include <type_traits>

namespace sampik {

// Standard mode: MPI implementation decides whether outgoing messages will
// be buffered. Send operations can be started whether or not a matching
// receive has been started. They may complete before a matching receive is
// started. Standard mode is non-local: successful completion of the send
// operation may depend on the occurrence of a matching receive.
struct StandardCommMode {};

// Synchronous mode: Send operations complete successfully only if a matching
// receive is started, and the receive operation has started to receive the
// message sent.
struct SynchronousCommMode {};

// Ready mode: Send operations may be started only if the matching receive is
// already started.
struct ReadyCommMode {};

// Default mode: lets the user override the send operations behavior at
// compile-time. E.g., this can be set to mode "Synchronous" for debug
// builds by defining SAMPIK_FORCE_SYNCHRONOUS_MODE.
#ifdef SAMPIK_FORCE_SYNCHRONOUS_MODE
using DefaultCommMode = SynchronousCommMode;
#else
using DefaultCommMode = StandardCommMode;
#endif

template <typename T>
struct is_communication_mode : std::false_type {};

template <>
struct is_communication_mode<StandardCommMode> : std::true_type {};

template <>
struct is_communication_mode<SynchronousCommMode> : std::true_type {};

template <>
struct is_communication_mode<ReadyCommMode> : std::true_type {};

template <typename T>
inline constexpr bool is_communication_mode_v = is_communication_mode<T>::value;

template <typename T>
concept CommunicationMode = is_communication_mode_v<T>;

}; // namespace sampik
