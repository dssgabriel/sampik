Utilities
=========

Wait and test functions for checking/completing communications:

.. code-block:: cpp

	auto wait(Request& req) -> std::expected<T, E>;
	auto wait_all(std::span<Request&> req) -> std::expected<T, E>;

.. code-block:: cpp

	auto test(Request& req) -> std::optional<T>;
	auto test_all(std::span<Request&> req) -> std::optional<T>;

.. note:: The exact return type for these function is yet to be determined.
